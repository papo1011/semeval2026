import random
import numpy as np
import torch
import os
import logging


def set_global_seed(seed: int = 42) -> None:
    """Locks all random number generators for strict reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you ever use multi-GPU

    # OS environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # This guarantees identical operations on the tensor cores,
    # but can slow down training by ~5%. It is worth it for research.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.getLogger(__name__).info(
        f"Global seed locked to {seed}. Training is strictly reproducible."
    )
