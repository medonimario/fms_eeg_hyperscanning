"""Plot recorded partner-training histories without loading EEG or model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

COLORS = {"true": "#0072B2", "shuffled": "#D55E00"}
LABELS = {"true": "True-pair training", "shuffled": "Shuffled-pair training"}


def plot_run(run: Path, output: Path) -> list[Path]:
    """Save unsmoothed retrieval and geometry curves with explicit chance references.

    Supports adaptation.json and alignment.json, including a renamed shuffled history.
    All training diagnostics are the recorded fixed-gallery estimates, not all training batches.
    """
    reports = [path for path in (run / "adaptation.json", run / "alignment.json") if path.exists()]
    if len(reports) != 1:
        raise ValueError("Expected exactly one adaptation.json or alignment.json report.")
    report = json.loads(reports[0].read_text())
    histories, sources = {}, [reports[0]]
    for arm in COLORS:
        candidates = [path for path in (run / arm / "history.csv", run / arm / f"history_{arm}.csv") if path.exists()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one history CSV for {arm}, found {len(candidates)}.")
        frame = pd.read_csv(candidates[0]).sort_values("epoch")
        if frame.epoch.duplicated().any() or frame.empty:
            raise ValueError(f"Empty or duplicate epochs in {arm} history.")
        histories[arm] = frame
        sources.append(candidates[0])
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    chance = report["chance_mrr"]
    train_count = len(report.get("training_diagnostic_groups", []))
    train_label = f"Fixed training galleries ({train_count} dyads)" if train_count else "Fixed training galleries"
    paths = []

    def panel(ax: plt.Axes, column: str, title: str, ylabel: str, reference: float | None = None) -> None:
        """Draw two observed epoch series and an optional reference, with missing metrics explicit."""
        for arm, frame in histories.items():
            if column not in frame:
                raise ValueError(f"Missing {column} in {arm} history.")
            ax.plot(frame.epoch, frame[column], color=COLORS[arm], label=LABELS[arm], marker="o", ms=3, lw=1.8)
        if reference is not None:
            label = "Uniform scores" if column == "train_loss" else "Zero margin" if "margin" in column else "Chance"
            ax.axhline(reference, color="#666666", ls="--", lw=1, label=f"{label}: {reference:.3f}")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, loc="best")

    def save(fig: plt.Figure, name: str, note: str) -> None:
        """Export readable PNG and vector PDF versions with interpretation notes."""
        fig.text(0.5, 0.015, note, ha="center", fontsize=9, color="#444444")
        fig.tight_layout(rect=(0, 0.055, 1, 0.93), h_pad=2)
        for suffix in ("png", "pdf"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=170, facecolor="white")
            paths.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    title = (
        "Partner retrieval: held-out trials, familiar dyads"
        if report.get("validation_protocol") == "within-dyad"
        else "Partner retrieval: fitting versus transfer"
    )
    fig.suptitle(title, fontsize=17)
    panel(axes[0, 0], "mrr", "Validation: real partner correspondence", "MRR (higher is better)", chance)
    panel(axes[0, 1], "top1", "Validation: first-ranked partner", "Top-1 accuracy", report["chance_top1"])
    for column, ax in (("mrr", axes[0, 0]), ("top1", axes[0, 1])):
        for arm, frame in histories.items():
            epoch = report["arms"][arm]["best_epoch"]
            chosen = frame.loc[frame.epoch.eq(epoch), column]
            if len(chosen) != 1:
                raise ValueError(f"Selected epoch {epoch} missing from {arm} history.")
            ax.scatter([epoch], chosen, marker="*", s=150, color=COLORS[arm], edgecolor="white", zorder=4)
    panel(axes[1, 0], "train_true_mrr", "Training: real partner correspondence", "MRR (higher is better)", chance)
    panel(
        axes[1, 1],
        "train_objective_mrr",
        "Training: matches assigned by each objective",
        "MRR (higher is better)",
        chance,
    )
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.set_ylim(
            max(
                0,
                min(
                    chance,
                    min(
                        float(d[c].min())
                        for d in histories.values()
                        for c in ("mrr", "train_true_mrr", "train_objective_mrr")
                    ),
                )
                - 0.03,
            ),
            max(
                0.52,
                max(
                    float(d[c].max())
                    for d in histories.values()
                    for c in ("mrr", "train_true_mrr", "train_objective_mrr")
                )
                + 0.03,
            ),
        )
    save(fig, "retrieval", f"{train_label}. Stars: checkpoints selected by validation MRR. No smoothing or test data.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Optimization and validation embedding geometry", fontsize=17)
    panel(axes[0, 0], "train_loss", "Training minibatch loss", "Symmetric contrastive loss", np.log(report["pairs"]))
    panel(
        axes[0, 1], "validation_cosine_margin", "Validation positive minus negative similarity", "Mean cosine margin", 0
    )
    panel(
        axes[1, 0],
        "validation_normalized_std",
        "Validation: spread after L2 normalization",
        "Mean coordinate standard deviation",
    )
    panel(axes[1, 1], "validation_effective_rank", "Validation: within-person effective rank", "Effective rank")
    axes[1, 1].set_ylim(0, report["pairs"])
    save(
        fig,
        "diagnostics",
        f"Effective rank is bounded by {report['pairs'] - 1} in these galleries; it is not whole-dataset rank.",
    )
    (output / "sources.json").write_text(json.dumps({"sources": [str(p.resolve()) for p in sources]}, indent=2) + "\n")
    return paths


def main() -> None:
    """Render partner-training curves to an explicitly supplied local output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in plot_run(args.run, args.output):
        print(path)


if __name__ == "__main__":
    main()
