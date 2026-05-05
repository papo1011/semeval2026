import os
import torch
import logging
from pathlib import Path
from datetime import datetime


class CheckpointManager:
    def __init__(
        self,
        save_dir: str,
        run_id: str = "run",
        keep_top_k: int = 3,
        mode: str = "min",
        logger: logging.Logger = None,
    ):
        """
        mode: 'min' for minimizing loss, 'max' for maximizing accuracy/F1
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{run_id}_{timestamp}"

        self.save_dir = Path(save_dir)
        if keep_top_k > 1:
            self.checkpoints_dir = self.save_dir / "checkpoints" / self.run_id
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.keep_top_k = keep_top_k
        self.mode = mode
        self.best_checkpoints = []  # List of tuples: (metric_value, filepath)
        self.logger = logger or logging.getLogger(__name__)

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        metric: float,
        optimizer: torch.optim.Optimizer = None,
        train_loss: float = None,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric": metric,
            "train_loss": train_loss,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Always save the latest checkpoint for crash-recovery
        latest_path = self.save_dir / f"{self.run_id}_latest.pt"
        torch.save(checkpoint, latest_path)

        if self.keep_top_k <= 1:
            return

        # Determine if this is a Top-K model
        is_top_k = False
        if len(self.best_checkpoints) < self.keep_top_k:
            is_top_k = True
        else:
            worst_kept_metric = self.best_checkpoints[-1][0]
            if self.mode == "min" and metric < worst_kept_metric:
                is_top_k = True
            elif self.mode == "max" and metric > worst_kept_metric:
                is_top_k = True

        if is_top_k:
            best_path = (
                self.checkpoints_dir
                / f"{self.run_id}_epoch{epoch}_metric{metric:.4f}.pt"
            )
            torch.save(checkpoint, best_path)

            self.best_checkpoints.append((metric, best_path))
            # Sort the tracker. If min (loss), ascending. If max (acc), descending.
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

            self.logger.info(
                f"New Top-{self.keep_top_k} model saved! ({best_path.name})"
            )

            # Prune the worst checkpoint from the hard drive
            if len(self.best_checkpoints) > self.keep_top_k:
                worst_metric, worst_path = self.best_checkpoints.pop(-1)
                if worst_path.exists():
                    try:
                        os.remove(worst_path)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to delete old checkpoint {worst_path.name}: {e}"
                        )
