import argparse
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, f1_score

from task_A.src.dataset import TaskA_Dataset
from task_A.src.model import (
    UniXcoderSupCon,
    LinearClassifier,
    UniXcoderSupConClassifier,
)
from utils.logger import setup_global_logger
from utils.seed import set_global_seed
from utils.checkpoint import CheckpointManager


def extract_features(loader, encoder, device, use_amp, desc="Extracting"):
    """Runs the forward pass exactly ONCE and caches the vectors in RAM."""
    all_features = []
    all_labels = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            with torch.amp.autocast(enabled=use_amp, device_type=str(device)):
                features = encoder(input_ids, attention_mask)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features), torch.cat(all_labels)


def train_stage2(args):
    logger = setup_global_logger(args.log_dir, prefix="TaskA_Classifier_Stage2")
    set_global_seed(42)

    if args.use_amp and not torch.cuda.is_available():
        logger.warning("AMP requested but CUDA is not available. Disabling AMP.")
        args.use_amp = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hardware initialization: Training on {device}")

    # Load the FROZEN Stage 1 Encoder
    logger.info(f"Initializing Model: {args.model_name} (Frozen Stage 1 Encoder)")
    encoder = UniXcoderSupCon(model_name=args.model_name, projection_dim=128).to(device)

    logger.info(f"Loading Stage 1 weights from: {args.stage1_weights}")
    checkpoint = torch.load(args.stage1_weights, map_location=device)
    if "model_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Successfully loaded structured checkpoint (Epoch {checkpoint.get('epoch', 'N/A')})."
        )
    else:
        encoder.load_state_dict(checkpoint)
        logger.info("Successfully loaded raw state_dict.")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Load Datasets
    logger.info(
        f"Loading Train and Validation datasets from {args.train_data} and {args.val_data}..."
    )
    train_dataset = TaskA_Dataset(
        parquet_path=args.train_data,
        model_name=args.model_name,
        max_length=args.max_length,
        is_train=True,
        use_normalization=args.normalize,
    )
    val_dataset = TaskA_Dataset(
        parquet_path=args.val_data,
        model_name=args.model_name,
        max_length=args.max_length,
        is_train=False,
        use_normalization=args.normalize,
    )

    # We can use a massive batch size here because Stage 2 takes very little
    raw_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    raw_val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    logger.info(
        f"Datasets loaded. Train batches: {len(raw_train_loader)} | Val batches: {len(raw_val_loader)} | Workers: {args.num_workers} | Pin Memory: {args.pin_memory}"
    )

    logger.info("Phase 1: Pre-computing 128-D Features")
    train_feat, train_lbl = extract_features(
        raw_train_loader, encoder, device, args.use_amp, "Extracting Train"
    )
    val_feat, val_lbl = extract_features(
        raw_val_loader, encoder, device, args.use_amp, "Extracting Val"
    )

    # no longer need the massive UniXcoder model in VRAM!
    del encoder
    torch.cuda.empty_cache()
    logger.info("UniXcoder offloaded from VRAM. Caching complete.")

    # Create new, ultra-fast DataLoaders that only serve 128-D vectors
    fast_train_loader = DataLoader(
        TensorDataset(train_feat, train_lbl), batch_size=512, shuffle=True
    )
    fast_val_loader = DataLoader(
        TensorDataset(val_feat, val_lbl), batch_size=512, shuffle=False
    )

    # Initialize the Classifier
    logger.info("Initializing Model: Linear Classification Head")
    classifier = LinearClassifier(input_dim=128, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    # AMP Scaler for the linear layer
    scaler = torch.amp.GradScaler(enabled=args.use_amp)

    logger.info("Phase 2: Training Linear Classification")
    logger.info(f"AMP Enabled: {args.use_amp} | Eff. Batch Size: 512 (Fast RAM Loader)")

    ckpt_manager = CheckpointManager(
        save_dir=args.save_dir,
        run_id="Classifier_Stage2",
        keep_top_k=3,
        mode="max",  # We want the HIGHEST accuracy
        logger=logger,
    )

    # Training Loop
    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(
            fast_train_loader,
            total=len(fast_train_loader),
            desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
        )

        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)

            with torch.amp.autocast(enabled=args.use_amp, device_type=str(device)):
                logits = classifier(features)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Validation Loop
        classifier.eval()
        all_preds, all_labels_eval = [], []

        with torch.no_grad():
            for features, labels in fast_val_loader:
                features = features.to(device)
                with torch.amp.autocast(enabled=args.use_amp, device_type=str(device)):
                    logits = classifier(features)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels_eval.extend(labels.numpy())

        # Metrics
        acc = accuracy_score(all_labels_eval, all_preds)
        macro_f1 = f1_score(all_labels_eval, all_preds, average="macro")
        train_loss_avg = total_loss / len(fast_train_loader)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} Completed | Train Loss: {train_loss_avg:.4f} | Val Accuracy: {acc * 100:.2f}% | Val Macro F1: {macro_f1:.4f}"
        )

        ckpt_manager.save(
            epoch=epoch + 1,
            model=classifier,
            optimizer=optimizer,
            metric=macro_f1,
            train_loss=train_loss_avg,
        )

    # --- FINAL EVALUATION REPORT ---
    logger.info("=" * 50)
    logger.info("FINAL CLASSIFICATION REPORT")
    logger.info("=" * 50)

    with open(args.id_map, "r") as f:
        id_to_label = json.load(f)

    num_classes = len(id_to_label)
    target_names = [id_to_label[str(i)] for i in range(num_classes)]

    report = classification_report(
        all_labels_eval, all_preds, target_names=target_names, digits=4
    )
    logger.info("\n" + report)
    logger.info("=" * 50)

    logger.info("Model Fusion")

    # Initialize the empty master model
    unified_model = UniXcoderSupConClassifier(
        model_name=args.model_name, projection_dim=128
    )

    # Load the frozen Stage 1 weights
    ckpt1 = torch.load(args.stage1_weights, map_location="cpu")
    unified_model.encoder.load_state_dict(ckpt1.get("model_state_dict", ckpt1))
    logger.info(
        f"-> Stage 1 Encoder ({Path(args.stage1_weights).name}) loaded into Unified Model."
    )

    # Load the newly trained classifier weights (using the best state from CheckpointManager)
    best_metric, best_classifier_path = ckpt_manager.best_checkpoints[-1]
    ckpt2 = torch.load(best_classifier_path, map_location="cpu")

    unified_model.classifier.load_state_dict(ckpt2.get("model_state_dict", ckpt2))
    logger.info(
        f"-> Stage 2 Classifier ({best_classifier_path.name}) loaded into Unified Model."
    )

    # Save the fused artifact
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_manger = CheckpointManager(
        save_dir=args.save_dir,
        run_id="UniXcoderSupConClassifier",
        keep_top_k=1,
        mode="max",
        logger=logger,
    )

    model_manger.save(
        epoch=args.epochs,
        model=unified_model,
        optimizer=None,
        metric=best_metric,
        train_loss=train_loss_avg,
    )

    logger.info("SUCCESS: Production model automatically fused and saved.")
    logger.info("=" * 50)


if __name__ == "__main__":
    TASK_A_DIR = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()

    # Data & Paths
    parser.add_argument(
        "--train_data", type=str, default=str(TASK_A_DIR / "data" / "train.parquet")
    )
    parser.add_argument(
        "--val_data", type=str, default=str(TASK_A_DIR / "data" / "validation.parquet")
    )
    parser.add_argument("--stage1_weights", type=str, required=True)
    parser.add_argument(
        "--id_map", type=str, default=str(TASK_A_DIR / "id_to_label.json")
    )
    parser.add_argument("--save_dir", type=str, default=str(TASK_A_DIR / "weights"))
    parser.add_argument("--log_dir", type=str, default=str(TASK_A_DIR / "logs"))

    # Model & Strategy
    parser.add_argument(
        "--model_name", type=str, default="microsoft/unixcoder-base-nine"
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Hardware & Scaling Toggles
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Physical batch size."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="CPU cores for dataloading."
    )
    parser.add_argument(
        "--pin_memory", action="store_true", help="Speeds up CPU-to-GPU transfer."
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Enable Automatic Mixed Precision."
    )

    args = parser.parse_args()
    train_stage2(args)
