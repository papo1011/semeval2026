import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from task_A.src.model import UniXcoderSupConClassifier
from utils.logger import setup_global_logger
from utils.seed import set_global_seed

try:
    from task_A.src.dataset import normalize_code_ast
except ImportError:
    normalize_code_ast = None


def collate_fn(batch, tokenizer, max_length, use_normalization, logger):
    codes = [item["code"] for item in batch]
    ids = [item["ID"] for item in batch]

    if use_normalization:
        if normalize_code_ast is None:
            logger.warning(
                "Normalization requested but function not found! Proceeding without it."
            )
        else:
            codes = [normalize_code_ast(c) for c in codes]

    encodings = tokenizer(
        codes, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    encodings["ids"] = ids
    return encodings


@torch.no_grad()
def predict(args):
    logger = setup_global_logger(args.log_dir, prefix="TaskA_Inference")
    set_global_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hardware initialization: Inferencing on {device}")

    # Load Tokenizer & Unified Model
    logger.info(f"Loading Tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info(f"Loading Master Model from: {args.model_weights}")
    model = UniXcoderSupConClassifier(
        model_name=args.model_name, projection_dim=128, num_classes=2
    ).to(device)

    # Safely unpack the rich dictionary style if present
    checkpoint = torch.load(args.model_weights, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    if "metric" in checkpoint:
        logger.info(
            f"Model loaded successfully! Historical Validation Macro F1: {checkpoint['metric']:.4f}"
        )

    logger.info(f"Streaming dataset from {args.parquet_path}...")
    dataset = load_dataset("parquet", data_files=args.parquet_path, split="train")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda x: collate_fn(
            x, tokenizer, args.max_length, args.normalize, logger
        ),
    )

    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Inference")
    with open(args.output_path, "w") as f:
        f.write("ID,label\n")

        for batch in tqdm(dataloader, desc="Predicting", total=len(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Single, seamless forward pass using Tensor Cores
            with torch.amp.autocast(enabled=True, device_type=str(device)):
                logits = model(input_ids, attention_mask)

            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

            for i, id_ in enumerate(batch["ids"]):
                f.write(f"{id_},{pred_ids[i]}\n")

    logger.info(f"Predictions successfully saved to {args.output_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    TASK_A_DIR = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Run inference with Unified UniXcoder pipeline"
    )

    # Model Weights (Now just a single file!)
    parser.add_argument(
        "--model_weights",
        type=str,
        default=str(TASK_A_DIR / "weights" / "production_model.pt"),
    )
    parser.add_argument(
        "--model_name", type=str, default="microsoft/unixcoder-base-nine"
    )

    # Data Paths
    parser.add_argument(
        "--parquet_path", type=str, required=True, help="Path to test parquet file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(TASK_A_DIR / "submission" / "predictions.csv"),
    )
    # parser.add_argument(
    #     "--id_map", type=str, default=str(TASK_A_DIR / "data" / "id_to_label.json")
    # )
    parser.add_argument("--log_dir", type=str, default=str(TASK_A_DIR / "logs"))

    # Settings
    parser.add_argument(
        "--normalize", action="store_true", help="Must match your training settings!"
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    predict(args)
