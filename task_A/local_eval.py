import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

from task_A.src.model import UniXcoderSupConClassifier
from utils.logger import setup_global_logger


def collate_fn(batch, tokenizer, max_length, use_normalization, normalize_fn):
    """Formats the batch. Normalization is passed explicitly to avoid global fallback scope."""
    codes = [item["code"] for item in batch]
    labels = [item["label"] for item in batch]

    if use_normalization:
        # We know normalize_fn is valid here because we failed-fast in main()
        codes = [normalize_fn(c) for c in codes]

    encodings = tokenizer(
        codes, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    encodings["labels"] = torch.tensor(labels, dtype=torch.long)
    return encodings


@torch.no_grad()
def evaluate(args):
    logger = setup_global_logger(args.log_dir, prefix=f"Task{args.task}_Evaluator")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 1. FAIL-FAST INTEGRITY CHECKS
    # ---------------------------------------------------------

    # Check Weights
    if not Path(args.model_weights).exists():
        logger.error(f"CRITICAL: Model weights not found at {args.model_weights}")
        raise FileNotFoundError(f"Missing weights file: {args.model_weights}")

    # Check Parquet Dataset
    if not Path(args.parquet_path).exists():
        logger.error(f"CRITICAL: Evaluation dataset not found at {args.parquet_path}")
        raise FileNotFoundError(f"Missing dataset file: {args.parquet_path}")

    # Check ID Map
    if not Path(args.id_map).exists():
        logger.error(f"CRITICAL: id_to_label.json not found at {args.id_map}")
        raise FileNotFoundError(
            "Label mapping JSON is required to determine the number of classes."
        )

    # Check Normalization Function
    normalize_code_ast = None
    if args.normalize:
        try:
            from task_A.src.dataset import normalize_code_ast

            logger.info("AST Normalization enabled and successfully imported.")
        except ImportError as e:
            logger.error(
                "CRITICAL: Normalization requested but 'normalize_code_ast' could not be imported."
            )
            raise e

    # ---------------------------------------------------------
    # 2. DYNAMIC SETUP
    # ---------------------------------------------------------

    logger.info(f"Loading label map from {args.id_map}...")
    with open(args.id_map, "r") as f:
        id_to_label = json.load(f)

    num_classes = len(id_to_label)
    target_names = [f"{label} ({idx})" for idx, label in id_to_label.items()]
    logger.info(f"Task configured strictly for {num_classes} classes.")

    # ---------------------------------------------------------
    # 3. MODEL INITIALIZATION
    # ---------------------------------------------------------

    logger.info(f"Loading Tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info(f"Initializing Model Architecture: {args.architecture}")

    match args.architecture:
        case "hf_baseline":
            # Load a standard HuggingFace Sequence Classifier (like CodeBERT)
            from transformers import AutoModelForSequenceClassification

            # Note: For HF baselines, model_weights is usually the directory saved by the Trainer
            logger.info(
                f"Loading HuggingFace pretrained weights from: {args.model_weights}"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_weights, num_labels=num_classes
            )

        case "supcon":
            # Load your custom Two-Stage SupCon Architecture
            model = UniXcoderSupConClassifier(
                model_name=args.model_name, projection_dim=128, num_classes=num_classes
            )
            logger.info(f"Loading strict model state from: {args.model_weights}")
            checkpoint = torch.load(args.model_weights, map_location=device)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

        case _:
            logger.error(
                f"CRITICAL: Unknown architecture requested: {args.architecture}"
            )
            raise ValueError(f"Architecture '{args.architecture}' is not supported.")

    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # 4. DATA LOADING & INFERENCE
    # ---------------------------------------------------------

    logger.info(f"Streaming evaluation dataset from {args.parquet_path}...")
    dataset = load_dataset("parquet", data_files=str(args.parquet_path), split="train")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda x: collate_fn(
            x, tokenizer, args.max_length, args.normalize, normalize_code_ast
        ),
    )

    all_preds = []
    all_labels = []

    logger.info("Starting Local Inference...")
    for batch in tqdm(dataloader, desc=f"Evaluating Task {args.task}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].numpy()

        with torch.amp.autocast(enabled=True, device_type=str(device)):
            logits = model(input_ids, attention_mask)

        # Apply Softmax to get real probabilities (0.0 to 1.0)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = (probs[:, 1] > args.threshold).long().cpu().numpy()

        all_preds.extend(pred_ids)
        all_labels.extend(labels)

    # ---------------------------------------------------------
    # 5. METRICS REPORTING
    # ---------------------------------------------------------

    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    logger.info("\n" + "=" * 50)
    logger.info(f"🏆 TASK {args.task} LOCAL MACRO F1 SCORE: {macro_f1:.5f}")
    logger.info("=" * 50)

    report = classification_report(
        all_labels, all_preds, target_names=target_names, digits=4
    )
    logger.info("\n" + report)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    # We use a two-pass parser to allow dynamic pathing while keeping code clean
    parser = argparse.ArgumentParser(
        description="Strict Universal Local Evaluator", add_help=False
    )
    parser.add_argument("-t", "--task", type=str, choices=["A", "B", "C"], default="A")

    temp_args, _ = parser.parse_known_args()
    TASK_DIR = BASE_DIR / f"task_{temp_args.task}"

    # Main parser with Short-Flags
    main_parser = argparse.ArgumentParser(parents=[parser])

    main_parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="HuggingFace model identifier (Default: CodeBERT Baseline)",
    )

    main_parser.add_argument(
        "-w",
        "--model_weights",
        type=str,
        required=True,
        help="Strict path to the .pt weights file (Required)",
    )
    main_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=["hf_baseline", "supcon"],
        default="hf_baseline",
        help="The architectural class of the model being evaluated",
    )

    main_parser.add_argument(
        "-p",
        "--parquet_path",
        type=str,
        default=str(TASK_DIR / "data" / "test_sample.parquet"),
        help="Path to the .parquet dataset to evaluate against",
    )

    main_parser.add_argument(
        "-i",
        "--id_map",
        type=str,
        default=str(TASK_DIR / "id_to_label.json"),
        help="Path to the id_to_label JSON file",
    )

    main_parser.add_argument(
        "-d",
        "--log_dir",
        type=str,
        default=str(TASK_DIR / "logs"),
        help="Directory to save logs",
    )

    main_parser.add_argument(
        "-n", "--normalize", action="store_true", help="Enable AST normalization"
    )
    main_parser.add_argument(
        "-l", "--max_length", type=int, default=512, help="Max sequence length"
    )
    main_parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="Inference batch size"
    )
    main_parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5, 
        help="Minimum confidence required to guess Machine (Default: 0.5)"
    )

    args = main_parser.parse_args()
    evaluate(args)
