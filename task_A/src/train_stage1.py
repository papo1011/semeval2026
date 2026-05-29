import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

from task_A.src.dataset import TaskA_Dataset
from task_A.src.model import UniXcoderSupCon
from task_A.src.losses import SupConLoss
from utils.logger import setup_global_logger
from utils.seed import set_global_seed
from utils.checkpoint import CheckpointManager


def train_stage1(args):
    # Initialize the global logger
    logger = setup_global_logger(args.log_dir, prefix="TaskA_SupCon_Stage1")
    set_global_seed(42)
    logger.info("Starting training pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hardware initialization: Training on {device}")
    if args.use_amp and not torch.cuda.is_available():
        logger.warning("AMP was requested but CUDA is not available. Disabling AMP.")
        args.use_amp = False

    # Load Datasets
    logger.info(f"Loading dataset from {args.train_data}...")
    train_dataset = TaskA_Dataset(
        parquet_path=args.train_data,
        # label_mapping_path=args.label_map,
        model_name=args.model_name,
        max_length=args.max_length,
        is_train=True,
        use_normalization=args.normalize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    logger.info(
        f"Dataset loaded. Total batches: {len(train_loader)} | Workers: {args.num_workers} | Pin Memory: {args.pin_memory}"
    )

    # Initialize Model & Loss
    logger.info(f"Initializing Model: {args.model_name}")
    model = UniXcoderSupCon(model_name=args.model_name, projection_dim=128).to(device)
    criterion = SupConLoss(temperature=args.temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # AMP SCALER (Dynamically enabled/disabled based on CLI flag)
    scaler = torch.amp.GradScaler(enabled=args.use_amp)

    logger.info("--- Starting Stage 1: Representation Learning ---")
    logger.info(
        f"AMP Enabled: {args.use_amp} | Eff. Batch Size: {args.batch_size * args.accumulation_steps}"
    )

    ckpt_manager = CheckpointManager(
        save_dir=args.save_dir,
        run_id="SupCon_Stage1",
        keep_top_k=3,
        mode="min",  # We want the LOWEST loss
        logger=logger,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Calculate total training steps
    total_steps = len(train_loader) * args.epochs
    # Warmup for the first 10% of training
    warmup_steps = int(0.1 * total_steps)

    # Create the scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Load tokenizer to get correct MASK and PAD token IDs
    logger.info("Loading tokenizer for Syntax Masking...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 15% of tokens will be masked for the SupCon augmentation
    masking_prob = 0.50

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            prob_matrix = torch.full(input_ids.shape, masking_prob).to(device)

            # Do NOT mask the padding tokens!
            prob_matrix.masked_fill_(input_ids == tokenizer.pad_token_id, 0.0)

            mask_indices = torch.bernoulli(prob_matrix).bool()

            # Replace the chosen tokens with the official [MASK] token
            input_ids[mask_indices] = tokenizer.mask_token_id

            optimizer.zero_grad(set_to_none=True)

            # AUTOCAST (Dynamically enabled/disabled based on CLI flag)
            with torch.amp.autocast(enabled=args.use_amp, device_type=str(device)):
                features = model(input_ids, attention_mask)
                loss = criterion(features, labels)
                # loss = loss / args.accumulation_steps

            # Backward pass via scaler (handles both AMP and standard FP32 safely)
            scaler.scale(loss).backward()

            # if (step + 1) % args.accumulation_steps == 0:
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            #     scaler.step(optimizer)
            #     scheduler.step()
            #     scaler.update()
            #     optimizer.zero_grad()

            # total_loss += loss.item() * args.accumulation_steps
            # progress_bar.set_postfix(
            #     {"Loss": f"{loss.item() * args.accumulation_steps:.4f}"}
            # )

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} Completed | Avg SupCon Loss: {avg_loss:.4f}"
        )

        # Call the manager at the end of the epoch!
        ckpt_manager.save(
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            metric=avg_loss,  # Passing our SupCon loss
            train_loss=avg_loss,
        )

    logger.info("Training Stage 1 Complete!")


if __name__ == "__main__":
    TASK_A_DIR = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()

    # Data & Paths
    parser.add_argument(
        "--train_data", type=str, default=str(TASK_A_DIR / "data" / "train.parquet")
    )
    parser.add_argument("--save_dir", type=str, default=str(TASK_A_DIR / "weights"))
    parser.add_argument("--log_dir", type=str, default=str(TASK_A_DIR / "logs"))

    # Model & Strategy
    parser.add_argument(
        "--model_name", type=str, default="microsoft/unixcoder-base-nine"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Pass this flag to enable Lexical Ablation",
    )
    parser.add_argument("--max_length", type=int, default=1024)

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)

    # Hardware & Scaling Toggles
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Physical batch size."
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=4, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="CPU cores for dataloading."
    )
    parser.add_argument(
        "--pin_memory", action="store_true", help="Speeds up CPU-to-GPU transfer."
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable Automatic Mixed Precision (Tensor Cores).",
    )

    args = parser.parse_args()
    train_stage1(args)
