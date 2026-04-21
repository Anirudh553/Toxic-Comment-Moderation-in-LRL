"""Training entry points and helpers."""

from src.training.train import train_baseline
from src.training.train_transformer import train_transformer

__all__ = ["train_baseline", "train_transformer"]
