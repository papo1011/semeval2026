import argparse
import logging
import os
import warnings
import inspect

import numpy as np
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
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


def pick_eval_key():
    sig = inspect.signature(TrainingArguments.__init__).parameters
    return "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"


class MyTrainer:
    def __init__(self, task_subset='C', max_length=512, model_name="bigcode/starcoder2-3b"):
        # map "C" -> "task_c"
        self.task_subset = task_subset
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.num_labels = None

    def load_and_prepare_data(self):
        logger.info(f"Loading dataset subset {self.task_subset} ...")
        dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.task_subset)

        train_data = dataset["train"]
        val_data = dataset["validation"]

        logger.info(f"Official Training samples: {len(train_data)}")
        logger.info(f"Official Validation samples: {len(val_data)}")

        train_df = train_data.to_pandas()
        val_df = val_data.to_pandas()

        def clean_df(df, name):
            if "code" not in df.columns or "label" not in df.columns:
                raise ValueError(f"{name} must contain 'code' and 'label' columns")
            df = df.dropna(subset=["code", "label"]).copy()
            df["label"] = df["label"].astype(int)
            return df

        train_df = clean_df(train_df, "Train")
        val_df = clean_df(val_df, "Validation")

        self.num_labels = int(train_df["label"].nunique())
        if self.task_subset == "task_c" and self.num_labels != 4:
            raise ValueError(f"Task C must have 4 labels, got {self.num_labels}")

        return train_df, val_df

    def initialize_model_and_tokenizer(self):
        logger.info(f"Initializing model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def tokenize_function(self, examples):
        
        return self.tokenizer(
            examples["code"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def prepare_datasets(self, train_df, val_df):
        train_dataset = Dataset.from_pandas(train_df[["code", "label"]])
        val_dataset = Dataset.from_pandas(val_df[["code", "label"]])

        train_dataset = train_dataset.map(self.tokenize_function, batched=True, remove_columns=["code"])
        val_dataset = val_dataset.map(self.tokenize_function, batched=True, remove_columns=["code"])

        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")

        train_dataset.set_format("torch")
        val_dataset.set_format("torch")
        return train_dataset, val_dataset

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds = np.argmax(predictions, axis=1)

        acc = accuracy_score(labels, preds)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    def train(self, train_dataset, val_dataset, output_dir, num_epochs=1, batch_size=2, learning_rate=2e-4):
        logger.info("Starting training...")

        eval_key = pick_eval_key()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=50,

            **{eval_key: "steps"},
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,

            learning_rate=learning_rate,
            optim="paged_adamw_32bit",

            bf16=True,
            fp16=False,
            tf32=True,

            gradient_checkpointing=True,
            report_to="none",
            save_total_limit=2,
            save_safetensors=True,
            dataloader_num_workers=2,
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
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return trainer

    def run_full_pipeline(self, output_dir, num_epochs=1, batch_size=2, learning_rate=2e-4):
        train_df, val_df = self.load_and_prepare_data()
        self.initialize_model_and_tokenizer()
        train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)
        self.train(train_dataset, val_dataset, output_dir, num_epochs, batch_size, learning_rate)


def main():
    parser = argparse.ArgumentParser(description="Task C Fine-tune (same style as Task A)")
    parser.add_argument("--model_name", default="bigcode/starcoder2-3b")
    parser.add_argument("--task", default="C")
    parser.add_argument("--output_dir", default="./results_taskc")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    trainer = MyTrainer(task_subset=args.task,
                         max_length=args.max_length,
                           model_name=args.model_name)
    
    trainer.run_full_pipeline(output_dir=args.output_dir,
                               num_epochs=args.epochs,
                                 batch_size=args.batch_size,
                                   learning_rate=args.lr)


if __name__ == "__main__":
    main()
