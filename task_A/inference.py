import argparse
import logging
import os
import warnings

import pandas as pd
import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

		self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		# Load Base Model in BF16 (matching training config)
		base_model = AutoModelForSequenceClassification.from_pretrained(
			base_model_name,
			num_labels=2,
			device_map=None,
			trust_remote_code=True,
			dtype=torch.bfloat16,
			attn_implementation="sdpa"
		)
		base_model.config.pad_token_id = self.tokenizer.pad_token_id

		# Merge LoRA adapters
		self.model = PeftModel.from_pretrained(base_model, self.model_dir)
		self.model.to(self.device)
		self.model.eval()

	def load_test_data(self):
		logger.info(f"Loading test dataset ...")
		current_dir = os.path.dirname(os.path.abspath(__file__))
		test_path = os.path.join(current_dir, "test.parquet")

		df = pd.read_parquet(test_path)
		return df

	def predict(self, df, batch_size):
		logger.info("Starting inference...")
		codes = df['code'].tolist()
		all_preds = []

		for i in tqdm(range(0, len(codes), batch_size)):
			batch_codes = codes[i: i + batch_size]

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
				all_preds.extend(preds.cpu().numpy())

		return all_preds

	def run(self, task_subset='A', batch_size=32, output_file="submission.csv"):
		self.load_model()
		test_df = self.load_test_data()
		predictions = self.predict(test_df, batch_size)

		submission = pd.DataFrame({
			'ID': test_df['ID'],
			'label': predictions
		})

		submission.to_csv(output_file, index=False)
		logger.info(f"Saved results to {output_file}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', type=str, required=True, help="Path to checkpoint folder")
	parser.add_argument('--output_file', default='submission.csv')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--max_length', type=int, default=512)

	args = parser.parse_args()

	inference = MyInference(
		model_dir=args.model_dir,
		max_length=args.max_length
	)

	inference.run(
		batch_size=args.batch_size,
		output_file=args.output_file
	)


if __name__ == "__main__":
	main()
