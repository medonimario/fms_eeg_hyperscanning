"""Read-only epoch-end evaluation and durable local learning curves."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from skorch.callbacks import Callback
from skorch.dataset import unpack_data
from torch.nn import functional
from torch.utils.data import Dataset

CURVE_KEYS = (
    "train_eval_loss",
    "train_eval_acc",
    "train_eval_balanced_accuracy",
    "valid_loss",
    "valid_acc",
    "valid_balanced_accuracy",
    "test_loss",
    "test_acc",
    "test_balanced_accuracy",
)


def evaluate_partition(net: Any, dataset: Dataset) -> dict[str, float]:
    """Evaluate all windows without gradients, weight updates, or RNG side effects.

    Args:
        net: Initialized classifier with a single-logit binary model.
        dataset: Complete partition, evaluated using the validation loader settings.

    Returns:
        Window-weighted BCE loss, accuracy, and balanced accuracy.
    """
    states = [(module, module.training) for module in net.module_.modules()]
    device = torch.device(net.device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    loss_sum, count = 0.0, 0
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            net.module_.eval()
            for batch in net.get_iterator(dataset, training=False):
                signal, target = unpack_data(batch)
                logits = net.infer(signal).float().reshape(-1)
                target = target.to(logits.device).reshape(-1)
                if not torch.isfinite(logits).all() or not ((target == 0) | (target == 1)).all():
                    raise ValueError("Expected finite binary logits and labels during evaluation.")
                loss_sum += functional.binary_cross_entropy_with_logits(logits, target.float(), reduction="sum").item()
                predicted = (logits > 0).long()
                confusion += torch.bincount((2 * target.long() + predicted).cpu(), minlength=4).reshape(2, 2)
                count += len(target)
    finally:
        for module, state in states:
            module.training = state
    totals = confusion.sum(dim=1)
    if count == 0 or (totals == 0).any():
        raise ValueError("Epoch evaluation requires both classes in a nonempty partition.")
    return dict(
        loss=loss_sum / count,
        acc=confusion.diag().sum().item() / count,
        balanced_accuracy=(confusion.diag().double() / totals).mean().item(),
    )


class EpochDiagnostics(Callback):
    """Measure train/test curves after validation selection; persist every completed epoch."""

    def __init__(self, output: Path, test: Dataset | None = None) -> None:
        """Retain the output directory and optional diagnostic test partition."""
        self.output = output
        self.test = test

    def on_epoch_end(self, net: Any, dataset_train: Dataset, **kwargs: Any) -> None:
        """Log evaluation-mode metrics without changing validation checkpoint decisions."""
        start = time.perf_counter()
        for prefix, dataset in (("train_eval", dataset_train), ("test", self.test)):
            if dataset is not None:
                for key, value in evaluate_partition(net, dataset).items():
                    net.history.record(f"{prefix}_{key}", value)
        net.history.record("diagnostic_duration", time.perf_counter() - start)
        history_tmp = self.output / "history.json.tmp"
        net.history.to_file(str(history_tmp))
        history_tmp.replace(self.output / "history.json")
        columns = ("epoch", "train_loss", *CURVE_KEYS, "dur", "diagnostic_duration")
        rows = [{key: row[key] for key in columns if key in row} for row in net.history]
        csv_tmp = self.output / "epoch_metrics.csv.tmp"
        pd.DataFrame(rows).to_csv(csv_tmp, index=False)
        csv_tmp.replace(self.output / "epoch_metrics.csv")
