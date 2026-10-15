"""Publication learning curves from complete, comparable cross-validation histories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .monitoring import CURVE_KEYS
from .runner import summarize


def collect_curves(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate ten complete runs and aggregate equally weighted folds with sample SD.

    Args:
        directory: Experiment directory containing fold_0 through fold_9.

    Returns:
        Long-form fold measurements and epoch/split/metric means and sample SDs.

    Raises:
        ValueError: If histories are missing metrics, incomplete, or non-finite.
    """
    rows = []
    expected_epochs = None
    for fold in range(10):
        folder = directory / f"fold_{fold}"
        config = json.loads((folder / "config.json").read_text())
        if config.get("model") == "logistic_regression":
            raise ValueError("Classical baselines have no epoch curves; use fold_metrics.csv and summary.json.")
        history = json.loads((folder / "history.json").read_text())
        epochs = [row["epoch"] for row in history]
        if epochs != list(range(1, config["options"]["epochs"] + 1)):
            raise ValueError(f"Fold {fold} has incomplete or duplicate epochs.")
        if expected_epochs is not None and epochs != expected_epochs:
            raise ValueError("All folds must have matching epochs.")
        expected_epochs = epochs
        for row in history:
            missing = set(CURVE_KEYS) - row.keys()
            if missing:
                raise ValueError(
                    f"Fold {fold}, epoch {row['epoch']} lacks {sorted(missing)}. "
                    "Older runs cannot supply unrecorded curves; rerun with epoch diagnostics enabled."
                )
            for split, prefix in (("train", "train_eval"), ("validation", "valid"), ("test", "test")):
                for metric, suffix in (
                    ("loss", "loss"),
                    ("accuracy", "acc"),
                    ("balanced_accuracy", "balanced_accuracy"),
                ):
                    value = float(row[f"{prefix}_{suffix}"])
                    if not np.isfinite(value):
                        raise ValueError(f"Non-finite curve value in fold {fold}.")
                    rows.append(dict(fold=fold, epoch=row["epoch"], split=split, metric=metric, value=value))
    summarize(directory)
    values = pd.DataFrame(rows)
    summary = (
        values.groupby(["epoch", "split", "metric"], sort=True)
        .value.agg(mean="mean", sd="std", n_folds="count")
        .reset_index()
    )
    return values, summary


def plot_curves(directory: Path, destination: Path | None = None) -> Path:
    """Export train/validation/test mean ± SD curves as PDF, SVG, PNG, and CSV."""
    values, summary = collect_curves(directory)
    destination = destination or directory / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    values.to_csv(destination / "learning_curves_folds.csv", index=False)
    summary.to_csv(destination / "learning_curves_summary.csv", index=False)
    figure = Figure(figsize=(14, 4.6), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    labels = {"loss": "Binary cross-entropy", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy"}
    colors = {"train": "#0072B2", "validation": "#009E73", "test": "#D55E00"}
    for axis, (metric, label) in zip(axes, labels.items(), strict=True):
        for split, color in colors.items():
            data = summary[(summary.metric == metric) & (summary.split == split)].sort_values("epoch")
            x, mean, sd = (data[column].to_numpy() for column in ("epoch", "mean", "sd"))
            axis.plot(x, mean, color=color, label=split.capitalize(), linewidth=2)
            axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.17, linewidth=0)
        axis.set(xlabel="Epoch", ylabel=label, title=label)
        axis.set_xticks(sorted(values.epoch.unique()))
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.suptitle(
        "10-fold learning curves: mean ± sample SD\nTrain evaluated at epoch end; test curves are diagnostic only",
        fontsize=12,
    )
    for extension in ("pdf", "svg", "png"):
        figure.savefig(destination / f"learning_curves.{extension}", dpi=300)
    figure.clear()
    return destination
