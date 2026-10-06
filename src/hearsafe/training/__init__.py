"""Local ESC-50 research training. PyTorch is imported only when invoked."""

from pathlib import Path
from typing import Any


def train(
    dataset: Path,
    output: Path,
    epochs: int = 50,
    patience: int = 8,
    batch_size: int = 32,
    seed: int = 42,
    threads: int = 4,
) -> dict[str, Any]:
    from .workflow import train as _train

    return _train(dataset, output, epochs, patience, batch_size, seed, threads)


def evaluate(dataset: Path, modeldir: Path, fold: int = 5) -> dict[str, Any]:
    from .workflow import evaluate as _evaluate

    return _evaluate(dataset, modeldir, fold)


__all__ = ["evaluate", "train"]
