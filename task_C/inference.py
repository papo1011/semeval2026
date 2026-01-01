import argparse
import logging
import os
import warnings

import pandas as pd
import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyInference:
    def __init__(self, model_dir, max_length=512):
        self.model_dir = model_dir
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_model(self):
        logger.info(f"Loading LoRA config from {self.model_dir}...")

        # Load adapter config to identify the base model
        config = PeftConfig.from_pretrained(self.model_dir)
        base_model_name = config.base_model_name_or_path
        logger.info(f"Base model: {base_model_name}")

        # Tokenizer (keep same style as Task A)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT: Task C has 4 labels
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=4,
            device_map=None,
            trust_remote_code=True,
            dtype=torch.float16,
            attn_implementation="sdpa"
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Merge LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def load_test_data(self):
        logger.info("Loading official test split from HuggingFace...")
        dataset = load_dataset("DaniilOr/SemEval-2026-Task13", "C")

        # The official test should have an ID column
        test_df = dataset["test"].to_pandas()

        if "id" in test_df.columns:
            id_col = "id"
        elif "ID" in test_df.columns:
            id_col = "ID"
        else:
            raise ValueError(f"Official test split has no id column. Found: {test_df.columns.tolist()}")

        if "code" not in test_df.columns:
            raise ValueError(f"Official test split has no code column. Found: {test_df.columns.tolist()}")

        return test_df, id_col

    def predict(self, df, batch_size):
        logger.info(f"Starting inference on {len(df)} samples...")
        codes = df["code"].tolist()
        all_preds = []

        for i in tqdm(range(0, len(codes), batch_size), desc="Predicting"):
            batch_codes = codes[i:i + batch_size]

            inputs = self.tokenizer(
                batch_codes,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())

        return all_preds

    def run(self, batch_size=32, output_file="submission.csv"):
        logger.info(f"Using device: {self.device}")
        self.load_model()

        test_df, id_col = self.load_test_data()
        predictions = self.predict(test_df, batch_size)

        # IMPORTANT: Output must be id,label (spec)
        submission = pd.DataFrame({
            "id": test_df[id_col],
            "label": predictions
        })

        submission.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        logger.info(f"First 5 rows:\n{submission.head(5)}")
        logger.info(f"Unique labels: {sorted(submission['label'].unique())}")
        logger.info(f"Total rows: {len(submission)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to checkpoint folder (LoRA adapter folder)")
    parser.add_argument("--output_file", default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    inference = MyInference(model_dir=args.model_dir, max_length=args.max_length)
    inference.run(batch_size=args.batch_size, output_file=args.output_file)


if __name__ == "__main__":
    main()
