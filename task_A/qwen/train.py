import argparse
import logging
import os
import warnings

import numpy as np
import torch
from datasets import load_dataset, Dataset
from peft import (
	LoraConfig,
	get_peft_model,
	TaskType
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	TrainingArguments,
	Trainer,
	EarlyStoppingCallback
)

warnings.filterwarnings("ignore")

# logging setup
local_rank = int(os.environ.get("LOCAL_RANK", -1))
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
)
logger = logging.getLogger(__name__)


class QwenTrainer:
	def __init__(self, task_subset='A', max_length=512, model_name="Qwen/Qwen2.5-Coder-14B"):
		self.task_subset = task_subset
		self.max_length = max_length
		self.model_name = model_name
		self.tokenizer = None
		self.model = None
		self.num_labels = None

	def load_and_prepare_data(self):
		if local_rank in [-1, 0]:
			logger.info(f"Loading dataset subset {self.task_subset}...")

		try:
			dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.task_subset)
			train_data = dataset['train']
			val_data = dataset['validation']

			if local_rank in [-1, 0]:
				logger.info(f"Official Training samples: {len(train_data)}")
				logger.info(f"Official Validation samples: {len(val_data)}")

			train_df = train_data.to_pandas()
			val_df = val_data.to_pandas()

			def clean_df(df):
				if 'code' not in df.columns or 'label' not in df.columns:
					raise ValueError(f"Dataset must contain 'code' and 'label' columns")
				df = df.dropna(subset=['code', 'label'])
				df['label'] = df['label'].astype(int)
				return df

			train_df = clean_df(train_df)
			val_df = clean_df(val_df)

			self.num_labels = train_df['label'].nunique()
			return train_df, val_df

		except Exception as e:
			logger.error(f"Error loading dataset: {e}")
			raise

	def initialize_model_and_tokenizer(self):
		if local_rank in [-1, 0]:
			logger.info(f"Initializing {self.model_name} in NATIVE BF16 for DDP")

		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		# This will occupy approx ~30GB of VRAM just for weights.
		# The A40 has 48GB, so this fits comfortably without quantization.
		self.model = AutoModelForSequenceClassification.from_pretrained(
			self.model_name,
			num_labels=self.num_labels,
			trust_remote_code=True,
			dtype=torch.bfloat16,
			attn_implementation="sdpa"  # Use native PyTorch SDPA
		)

		self.model.config.pad_token_id = self.tokenizer.pad_token_id
		self.model.config.use_cache = False

		# CRITICAL: Without this, VRAM usage would double during backprop (30GB -> 60GB), causing OOM.
		# This trades a small amount of compute speed for massive memory savings.
		self.model.gradient_checkpointing_enable()

		# Target all linear layers for maximum performance on code tasks
		peft_config = LoraConfig(
			task_type=TaskType.SEQ_CLS,
			r=64,
			lora_alpha=128,
			lora_dropout=0.05,
			target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
			bias="none"
		)

		self.model = get_peft_model(self.model, peft_config)

		if local_rank in [-1, 0]:
			self.model.print_trainable_parameters()

	def tokenize_function(self, examples):
		return self.tokenizer(
			examples['code'],
			truncation=True,
			padding="max_length",
			max_length=self.max_length,
			return_tensors="pt"
		)

	def prepare_datasets(self, train_df, val_df):
		train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
		val_dataset = Dataset.from_pandas(val_df[['code', 'label']])

		train_dataset = train_dataset.map(self.tokenize_function, batched=True, remove_columns=['code'])
		val_dataset = val_dataset.map(self.tokenize_function, batched=True, remove_columns=['code'])

		train_dataset = train_dataset.rename_column('label', 'labels')
		val_dataset = val_dataset.rename_column('label', 'labels')

		train_dataset.set_format("torch")
		val_dataset.set_format("torch")
		return train_dataset, val_dataset

	def compute_metrics(self, eval_pred):
		predictions, labels = eval_pred
		# Handle tuple outputs (common in LoRA models)
		if isinstance(predictions, tuple):
			predictions = predictions[0]
		predictions = np.argmax(predictions, axis=1)

		accuracy = accuracy_score(labels, predictions)
		# SemEval uses Macro F1
		precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
		return {'accuracy': accuracy, 'f1_macro': f1}

	def train(self, train_dataset, val_dataset, output_dir="./results_qwen", num_epochs=1, batch_size=8,
			  learning_rate=2e-4):

		if local_rank in [-1, 0]:
			logger.info("Starting training...")

		training_args = TrainingArguments(
			dataloader_num_workers=0,

			output_dir=output_dir,
			num_train_epochs=num_epochs,

			# Batch size per GPU. (es. 8 * 4 GPU = 32 global batch)
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			gradient_accumulation_steps=1,

			warmup_steps=100,
			weight_decay=0.01,
			logging_steps=10,

			eval_strategy="steps",
			eval_steps=500,
			save_strategy="steps",
			save_steps=500,
			load_best_model_at_end=True,
			metric_for_best_model="f1_macro",

			learning_rate=learning_rate,
			optim="paged_adamw_32bit",

			# Hardware Settings
			bf16=True,
			fp16=False,
			tf32=True,

			# DDP specific settings
			ddp_find_unused_parameters=False,

			gradient_checkpointing=True,
			report_to="none",
			save_total_limit=2,
			save_safetensors=True
		)

		trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=val_dataset,
			tokenizer=self.tokenizer,
			compute_metrics=self.compute_metrics,
			callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
		)

		trainer.train()

		if local_rank in [-1, 0]:
			trainer.save_model(output_dir)
			self.tokenizer.save_pretrained(output_dir)
			logger.info(f"Model saved to {output_dir}")

		return trainer

	def run_full_pipeline(self, output_dir, num_epochs, batch_size, learning_rate):
		try:
			train_df, val_df = self.load_and_prepare_data()
			self.initialize_model_and_tokenizer()
			train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)

			trainer = self.train(
				train_dataset, val_dataset,
				output_dir=output_dir,
				num_epochs=num_epochs,
				batch_size=batch_size,
				learning_rate=learning_rate
			)
			return trainer
		except Exception as e:
			logger.error(f"Error in pipeline: {e}")
			raise


def main():
	parser = argparse.ArgumentParser(description="Fine-tune Qwen 14B (Native BF16)")
	parser.add_argument('--task', default='A')
	parser.add_argument('--output_dir', default='./results_qwen')
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=8, help="Per-GPU batch size")
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--max_length', type=int, default=512)

	args = parser.parse_args()

	if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
		os.makedirs(args.output_dir, exist_ok=True)

	trainer = QwenTrainer(
		task_subset=args.task,
		max_length=args.max_length,
		model_name="Qwen/Qwen2.5-Coder-14B"
	)

	trainer.run_full_pipeline(
		output_dir=args.output_dir,
		num_epochs=args.epochs,
		batch_size=args.batch_size,
		learning_rate=args.lr
	)


if __name__ == "__main__":
	main()
