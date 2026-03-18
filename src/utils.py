import logging
import torch
from pathlib import Path


def get_device():
    """Return best available device.
    
    WHY this order? CUDA first for Nvidia GPUs,
    MPS second for Apple Silicon, CPU as fallback.
    Never hardcode .cuda() — this works on any machine.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str):
    """Return a logger that prints to terminal with timestamps.
    
    WHY not just print()?
    Logger gives timestamps and severity levels.
    Makes it easy to track how long each step takes.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


import random
import numpy as np

def set_seed(seed: int = 42):
    """Seed everything for reproducible runs.
    
    WHY seed all three? Each library has its own RNG.
    Seeding only torch won't fix numpy's shuffles.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)    