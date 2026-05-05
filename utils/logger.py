import logging
import os
from datetime import datetime


def setup_global_logger(save_dir, prefix):
    """
    Configures a dual-output logger (Console + File) to be used across all tasks.

    Args:
        save_dir (str): Where to save the log file.
        prefix (str): Identifies the task/script (e.g., 'taskA_stage1', 'taskB_baseline')
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"{prefix}_{timestamp}.log")

    # We clear existing handlers so we don't accidentally print twice if called multiple times
    logging.getLogger().handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    return logger
