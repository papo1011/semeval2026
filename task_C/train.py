import argparse
import os
import warnings
import inspect
import sys

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


def logx(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def pick_eval_key():
    sig = inspect.signature(TrainingArguments.__init__).parameters
    return "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"


class MyTrainer:
    def __init__(self, task_subset="C", max_length=128, model_name="bigcode/starcoder2-3b", smoke=True):
        self.task_subset = task_subset
        self.max_length = max_length
        self.model_name = model_name
        self.smoke = smoke

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.num_labels = None

    def load_and_prepare_data(self):
        logx(f">>> Loading dataset subset: {self.task_subset}")
        dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.task_subset)

        train_data = dataset["train"]
        val_data = dataset["validation"]

        logx(f">>> Official train size: {len(train_data)}")
        logx(f">>> Official validation size: {len(val_data)}")

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

        # SMOKE MODE: tiny subset to verify the pipeline quickly on Colab
        if self.smoke:
            train_df = train_df.sample(200, random_state=42)
            val_df = val_df.sample(50, random_state=42)
            logx(f">>> SMOKE ENABLED: train={len(train_df)} val={len(val_df)}")

        self.num_labels = int(train_df["label"].nunique())
        if str(self.task_subset).upper() == "C" and self.num_labels != 4:
            raise ValueError(f"Task C must have 4 labels, got {self.num_labels}")

        return train_df, val_df

    def initialize_model_and_tokenizer(self):
        logx(f">>> Initializing model: {self.model_name}")
        logx(f">>> Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # T4-friendly: fp16
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            trust_remote_code=True,
            dtype=torch.float16
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

        self.model.to(self.device)
        logx(">>> Model is ready on the selected device.")

    def tokenize_function(self, examples):
        # padding=True is faster than padding="max_length" for smoke / quick runs
        return self.tokenizer(
            examples["code"],
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )

    def prepare_datasets(self, train_df, val_df):
        logx(">>> Building Hugging Face datasets...")
        train_dataset = Dataset.from_pandas(train_df[["code", "label"]])
        val_dataset = Dataset.from_pandas(val_df[["code", "label"]])

        logx(">>> Tokenizing train split...")
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, remove_columns=["code"])

        logx(">>> Tokenizing validation split...")
        val_dataset = val_dataset.map(self.tokenize_function, batched=True, remove_columns=["code"])

        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")

        train_dataset.set_format("torch")
        val_dataset.set_format("torch")
        logx(">>> Tokenization completed.")
        return train_dataset, val_dataset

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds = np.argmax(predictions, axis=1)

        acc = accuracy_score(labels, preds)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    def train(self, train_dataset, val_dataset, output_dir, num_epochs=1, batch_size=1, learning_rate=2e-4):
        logx(">>> Creating TrainingArguments...")

        eval_key = pick_eval_key()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,

            # T4-friendly
            gradient_accumulation_steps=8,

            warmup_steps=20,
            weight_decay=0.01,
            logging_steps=5,

            **{eval_key: "steps"},
            eval_steps=20,
            save_strategy="steps",
            save_steps=20,

            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,

            learning_rate=learning_rate,
            optim="adamw_torch",

            bf16=False,
            fp16=True,
            tf32=False,

            gradient_checkpointing=True,
            report_to="none",
            save_total_limit=2,
            save_safetensors=True,

            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        logx(">>> ***** Running training *****")
        trainer.train()

        logx(">>> Saving model...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logx(f">>> Saved to: {output_dir}")

    def run_full_pipeline(self, output_dir, num_epochs=1, batch_size=1, learning_rate=2e-4):
        logx(">>> STEP 1: Load data")
        train_df, val_df = self.load_and_prepare_data()
        logx(f">>> STEP 1 DONE: train={len(train_df)} val={len(val_df)}")

        logx(">>> STEP 2: Initialize model")
        self.initialize_model_and_tokenizer()
        logx(">>> STEP 2 DONE")

        logx(">>> STEP 3: Tokenize datasets")
        train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)
        logx(">>> STEP 3 DONE")

        logx(">>> STEP 4: Train")
        self.train(train_dataset, val_dataset, output_dir, num_epochs, batch_size, learning_rate)
        logx(">>> STEP 4 DONE")


def main():
    parser = argparse.ArgumentParser(description="Task C Fine-tune (T4-safe + anti-Colab-kill logs)")
    parser.add_argument("--model_name", default="bigcode/starcoder2-3b")
    parser.add_argument("--task", default="C")
    parser.add_argument("--output_dir", default="./results_taskc_smoke")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--no_smoke", action="store_true", help="Disable smoke sampling (use full dataset)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = MyTrainer(
        task_subset=args.task,
        max_length=args.max_length,
        model_name=args.model_name,
        smoke=(not args.no_smoke)
    )

    trainer.run_full_pipeline(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
