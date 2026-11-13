"""Headless line plots for tiny-fit runs, without loading signals or model weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


def plot_tiny_history(history: pd.DataFrame, output: Path, chance: float) -> None:
    """Refresh loss and EEG-to-movement retrieval curves, including partially completed runs."""
    profiles = [p for p in ("pooled", "temporal", "full") if p in set(history.profile)]
    if not profiles or not 0 < chance <= 1:
        raise ValueError("Expected nonempty tiny-fit history and a valid chance value")
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(profiles), 2, figsize=(11, 2.8 * len(profiles)), squeeze=False, layout="constrained"
    )
    for index, profile in enumerate(profiles):
        left, right = axes[index]
        for arm, color in (("true", "#1675b6"), ("shuffled", "#df7d13")):
            rows = history[(history.profile == profile) & (history.arm == arm)].sort_values("step")
            if rows.empty:
                continue
            left.plot(rows.step, rows.loss.clip(lower=1e-12), color=color, label=f"{arm.capitalize()} pairing")
            right.plot(rows.step, rows.assigned_top1, color=color, label=f"{arm.capitalize()}: assigned matches")
            if arm == "shuffled":
                right.plot(
                    rows.step, rows.true_top1, color="#555555", linestyle="--", label="Shuffled: original matches"
                )
        left.set_yscale("log")
        left.set(title=f"{profile.capitalize()} — contrastive loss", ylabel="Loss (log scale)")
        right.axhline(chance, color="#888888", linestyle=":", label=f"Chance ({chance:.1%})")
        right.set(
            title=f"{profile.capitalize()} — EEG → movement top-1", ylabel="Retrieval accuracy", ylim=(-0.03, 1.05)
        )
        right.yaxis.set_major_formatter(PercentFormatter(1))
        for axis in (left, right):
            axis.set_xlabel("Optimizer step")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, loc="best")
    figure.suptitle("Tiny-fit: training-set memorization only", fontsize=14)
    for extension in ("png",):
        temporary = output / f".learning_curves.{extension}"
        figure.savefig(temporary, format=extension, dpi=150)
        temporary.replace(output / f"learning_curves.{extension}")
    plt.close(figure)


def plot_tiny_run(run: Path, output: Path) -> None:
    """Verify a completed history and export figures into a new directory."""
    metadata = json.loads((run / "tiny_fit.json").read_text())
    history_path = run / "history.csv"
    checksum = hashlib.sha256(history_path.read_bytes()).hexdigest()
    if metadata.get("operation") != "tiny-fit" or checksum != metadata["output_sha256"]["history.csv"]:
        raise ValueError("Invalid tiny-fit history or checksum")
    output.mkdir(parents=True, exist_ok=False)
    plot_tiny_history(pd.read_csv(history_path), output, metadata["chance_top1"])
    (output / "plots.json").write_text(
        json.dumps(
            dict(
                source=str(run.resolve()),
                history_sha256=checksum,
                tiny_fit_sha256=hashlib.sha256((run / "tiny_fit.json").read_bytes()).hexdigest(),
                scope="Training-only tiny-fit curves; no model loading or retraining",
            ),
            indent=2,
        )
        + "\n"
    )


def plot_training_history(history: pd.DataFrame, output: Path, arm: str) -> None:
    """Refresh PNG curves for assigned training loss and true-correspondence retrieval."""
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
    for split, color in (("train", "#1675b6"), ("validation", "#df7d13")):
        axes[0].plot(
            history.epoch,
            history[f"{split}_assigned_loss"],
            color=color,
            label=f"{split}: {'assigned' if split == 'train' else 'true'} pairs",
        )
        for axis, metric in zip(axes[1:], ("top1", "mrr"), strict=True):
            axis.plot(history.epoch, history[f"{split}_{metric}"], color=color, label=split)
            axis.plot(
                history.epoch, history[f"{split}_chance_{metric}"], color=color, linestyle=":", label=f"{split} chance"
            )
    axes[0].set(title="Symmetric contrastive loss", ylabel="Loss")
    axes[1].set(title="EEG → movement top-1 (true pairs)", ylabel="Accuracy", ylim=(-0.02, 1.02))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set(title="EEG → movement MRR (true pairs)", ylabel="MRR", ylim=(0, 1.02))
    for axis in axes:
        axis.set_xlabel("Epoch (0 = initialization)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(f"{output.parent.name.capitalize()} — {arm} training arm; dyad-macro scores")
    temporary = output / ".learning_curves.png"
    figure.savefig(temporary, format="png", dpi=150)
    temporary.replace(output / "learning_curves.png")
    plt.close(figure)


def plot_ridge_history(history: pd.DataFrame, output: Path) -> None:
    """Plot PNG-only validation retrieval against regularization for both ridge families."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
    for family, color in (("shared", "#1675b6"), ("participant", "#df7d13")):
        for arm, style in (("true", "-"), ("shuffled", "--")):
            rows = history[(history.family == family) & (history.arm == arm)].sort_values("regularization")
            if rows.empty:
                continue
            for axis, metric in zip(axes, ("top1", "mrr", "mse"), strict=True):
                axis.plot(
                    rows.regularization,
                    rows[f"validation_{metric}"],
                    color=color,
                    linestyle=style,
                    marker="o",
                    markersize=3,
                    label=f"{family}: {arm}",
                )
    for axis, metric in zip(axes, ("top1", "mrr", "mse"), strict=True):
        if metric != "mse":
            axis.axhline(history[f"validation_chance_{metric}"].iloc[0], color="gray", linestyle=":", label="Chance")
        else:
            axis.axhline(
                history.validation_zero_prediction_mse.iloc[0], color="gray", linestyle=":", label="Training mean"
            )
        axis.set_xscale("log")
        axis.set_xlabel("Regularization λ")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set(title="Validation EEG → movement top-1", ylabel="Accuracy")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(title="Validation MRR", ylabel="MRR")
    axes[2].set(title="Normalized movement prediction error", ylabel="MSE (lower is better)")
    figure.suptitle("Ridge diagnostic — dyad-macro validation scores; true correspondence")
    temporary = output / ".regularization_curves.png"
    figure.savefig(temporary, format="png", dpi=150)
    temporary.replace(output / "regularization_curves.png")
    plt.close(figure)
