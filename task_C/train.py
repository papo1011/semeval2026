import argparse
import logging
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_MAP = {"A": "task_a", "B": "task_b", "C": "task_c"}

def pick_id_key(keys):
    # طبق spec باید "id" باشه، ولی بعضی جاها "ID"
    if "id" in keys: return "id"
    if "ID" in keys: return "ID"
    return None

def collate_fn(batch, tokenizer, max_length, id_key):
    codes = [x["code"] for x in batch]
    ids = [x[id_key] for x in batch]
    enc = tokenizer(codes, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    enc["ids"] = ids
    return enc

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="path to trained model folder")
    ap.add_argument("--task", default="C")
    ap.add_argument("--split", default="test", help="train/validation/test")
    ap.add_argument("--output_csv", default="submission.csv")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda or cpu")
    args = ap.parse_args()

    task_subset = TASK_MAP.get(args.task.upper(), args.task)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, trust_remote_code=True)
    model.to(device)
    model.eval()

    dataset = load_dataset("DaniilOr/SemEval-2026-Task13", task_subset, split=args.split)
    first = dataset[0]
    if "code" not in first:
        raise ValueError("Dataset must contain 'code' column")

    id_key = pick_id_key(first.keys())
    if id_key is None:
        raise ValueError("Dataset must contain 'id' (or 'ID') column for submission")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, id_key),
    )

    rows = []
    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

        for i, _id in enumerate(batch["ids"]):
            rows.append((_id, preds[i]))

    out = pd.DataFrame(rows, columns=["id", "label"])
    out.to_csv(args.output_csv, index=False)
    logger.info(f"Saved: {args.output_csv}")

if __name__ == "__main__":
    main()
