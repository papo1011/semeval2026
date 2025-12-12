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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenLoRATrainer:
	def __init__(self, task_subset='A', max_length=512, model_name="Qwen/Qwen2.5-Coder-14B"):
		self.task_subset = task_subset
		self.max_length = max_length
		self.model_name = model_name
		self.tokenizer = None
		self.model = None
		self.num_labels = None

	def load_and_prepare_data(self):
		logger.info(f"Loading dataset subset {self.task_subset}...")
		try:
			# Load the official dataset
			dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.task_subset)
			train_data = dataset['train']
			val_data = dataset['validation']

			logger.info(f"Official Training samples: {len(train_data)}")
			logger.info(f"Official Validation samples: {len(val_data)}")

			train_df = train_data.to_pandas()
			val_df = val_data.to_pandas()

			def clean_df(df, name):
				if 'code' not in df.columns or 'label' not in df.columns:
					raise ValueError(f"{name} Dataset must contain 'code' and 'label' columns")

				# Drop NaNs
				original_len = len(df)
				df = df.dropna(subset=['code', 'label'])
				if len(df) < original_len:
					logger.warning(f"Dropped {original_len - len(df)} NaN rows from {name}")

				df['label'] = df['label'].astype(int)
				return df

			train_df = clean_df(train_df, "Train")
			val_df = clean_df(val_df, "Validation")

			self.num_labels = train_df['label'].nunique()
			logger.info(f"Number of unique labels: {self.num_labels}")

			return train_df, val_df

		except Exception as e:
			logger.error(f"Error loading dataset: {e}")
			raise

	def initialize_model_and_tokenizer(self):
		logger.info(f"Initializing {self.model_name} in NATIVE BFLOAT16 (No Quantization)...")

		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
		# Qwen often lacks a default pad_token; we set it to EOS
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		# We use torch_dtype=torch.bfloat16 to load in native 16-bit precision.
		# This requires ~28GB of VRAM for the weights alone.

		use_flash_attn = False
		if torch.cuda.get_device_capability()[0] >= 8:  # A100/H100 supports Flash Attn 2
			use_flash_attn = True

		self.model = AutoModelForSequenceClassification.from_pretrained(
			self.model_name,
			num_labels=self.num_labels,
			device_map="auto",
			trust_remote_code=True,
			torch_dtype=torch.bfloat16,
			attn_implementation="flash_attention_2" if use_flash_attn else "sdpa"
		)

		# Configuration adjustments for training
		self.model.config.pad_token_id = self.tokenizer.pad_token_id
		self.model.config.use_cache = False

		# Essential to prevent OOM errors with long sequences, even on A100 80GB
		self.model.gradient_checkpointing_enable()

		# Target all linear layers to maximize performance
		peft_config = LoraConfig(
			task_type=TaskType.SEQ_CLS,
			r=64,  # Rank
			lora_alpha=128,  # Scaling factor
			lora_dropout=0.05,
			target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
			bias="none"
		)

		self.model = get_peft_model(self.model, peft_config)
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
		# Convert Pandas DF to HF Dataset
		train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
		val_dataset = Dataset.from_pandas(val_df[['code', 'label']])

		# Apply tokenization
		train_dataset = train_dataset.map(self.tokenize_function, batched=True, remove_columns=['code'])
		val_dataset = val_dataset.map(self.tokenize_function, batched=True, remove_columns=['code'])

		# Rename columns for PyTorch compatibility
		train_dataset = train_dataset.rename_column('label', 'labels')
		val_dataset = val_dataset.rename_column('label', 'labels')

		# Set format to torch tensors
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
		# SemEval uses Macro F1 score
		precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
		return {'accuracy': accuracy, 'f1_macro': f1}

	def train(self, train_dataset, val_dataset, output_dir="./results_qwen", num_epochs=1, batch_size=8,
			  learning_rate=2e-4):
		logger.info("Starting training")

		training_args = TrainingArguments(
			output_dir=output_dir,
			num_train_epochs=num_epochs,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			gradient_accumulation_steps=2,
			warmup_steps=100,
			weight_decay=0.01,
			logging_steps=10,

			# Evaluation Strategy
			eval_strategy="steps",
			eval_steps=200,
			save_strategy="steps",
			save_steps=200,
			load_best_model_at_end=True,
			metric_for_best_model="f1_macro",

			learning_rate=learning_rate,
			optim="adamw_torch",  # Standard optimizer since we have enough memory

			# Hardware Optimization for A100
			bf16=True,  # Mandatory on A100 for stability with BF16 weights
			fp16=False,  # Disable standard FP16
			tf32=True,  # Enable TensorFloat32 (boosts matrix math speed)

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

		# Save final model
		trainer.save_model(output_dir)
		self.tokenizer.save_pretrained(output_dir)
		return trainer

	def run_full_pipeline(self, output_dir="./results_qwen", num_epochs=1, batch_size=8, learning_rate=2e-4):
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
	parser = argparse.ArgumentParser(description="Fine-tune Qwen 14B")
	parser.add_argument('--task', choices=['A', 'B', 'C'], default='A', help='SemEval Task subset')
	parser.add_argument('--output_dir', default='./results_qwen', help='Directory to save results')
	parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
	parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU (Try 16 on 80GB VRAM)')
	parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
	parser.add_argument('--max_length', type=int, default=1024, help='Max sequence length')

	args = parser.parse_args()
	os.makedirs(args.output_dir, exist_ok=True)

	trainer = QwenLoRATrainer(
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
