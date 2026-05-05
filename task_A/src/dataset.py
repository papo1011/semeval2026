import pandas as pd
import torch
import re
import logging
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def normalize_code_ast(code_str):
    """Lexical Ablation: Strips non-executable formatting."""
    code_str = str(code_str)
    code_str = re.sub(r'"""[\s\S]*?"""', "", code_str)
    code_str = re.sub(r"'''[\s\S]*?'''", "", code_str)
    code_str = re.sub(r"#.*", "", code_str)
    return "\n".join([line.strip() for line in code_str.split("\n") if line.strip()])


class TaskA_Dataset(Dataset):
    def __init__(
        self,
        parquet_path,
        model_name="microsoft/unixcoder-base-nine",
        max_length=1024,
        is_train=True,
        use_normalization=False,
        logger: logging.Logger = None,
    ):
        if logger:
            logger.info(f"Loading data from {parquet_path}...")
        self.data = pd.read_parquet(parquet_path)

        # Prevent SupCon data leakage
        if is_train:
            initial_len = len(self.data)
            self.data = self.data.drop_duplicates(subset=["code"]).reset_index(
                drop=True
            )
            if logger:
                logger.info(
                    f"Dropped {initial_len - len(self.data)} exact duplicates from training data."
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.use_normalization = use_normalization

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        raw_code = row["code"]

        label = int(row["label"])

        if self.use_normalization:
            code_text = normalize_code_ast(raw_code)
        else:
            code_text = raw_code

        inputs = self.tokenizer(
            code_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
