"""Locked temporal-head evaluation on original external-validation dyads only."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import sha256
from .overfit import ContrastiveLabram
from .ssl import EEGCache, SSLConfig
from .temporal import temporal_features, temporal_metrics, temporal_summary


class SavedStandardizer(nn.Module):
    """Restore training statistics without fitting any evaluation features."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        """Register immutable training-fitted statistics."""
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the saved transformation."""
        return (values - self.mean) / self.scale


def verified_file(root: Path, name: str, digest: str) -> Path:
    """Resolve and verify a file inside the completed source run."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
        raise ValueError(f"Source provenance mismatch: {name}")
    return path


def external_trials(cache: EEGCache, pairs: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select fixed paired galleries from clean offset intersections in fold one."""
    rows = pd.read_csv(cache.path / "windows.csv")
    rows = rows[rows.fold == 1].copy()
    keys = ["group", "condition", "trial_id"]
    a = rows[rows.crop_start_s == 0]
    b = rows[rows.crop_start_s == 4][keys + ["array", "array_index"]].rename(
        columns={"array": "target_array", "array_index": "target_array_index"}
    )
    joined = a.merge(b, on=keys, validate="one_to_one")
    census = rows.groupby(["group", "condition"]).trial_id.nunique().rename("cached_trials").to_frame()
    census = census.join(joined.groupby(["group", "condition"]).size().rename("eligible_trials")).fillna(0)
    census["eligible_trials"] = census.eligible_trials.astype(int)
    if census.empty or (census.eligible_trials < pairs).any():
        raise ValueError(f"Insufficient clean external temporal trials for {pairs} candidates:\n{census}")
    rng = np.random.default_rng(seed)
    selected = []
    for _, frame in joined.groupby(["group", "condition"], sort=True):
        frame = frame.sort_values("condition_trial")
        selected.append(frame.iloc[rng.choice(len(frame), pairs, replace=False)])
    return pd.concat(selected).assign(split="validation").reset_index(drop=True), census.reset_index()


def evaluate_transfer(
    cache_path: Path,
    run: Path,
    output: Path,
    checkpoint: Path,
    experiment: str = "temperature_0.05",
    device: str = "cpu",
) -> dict:
    """Evaluate one explicitly locked configuration without updating or selecting models."""
    report = json.loads((run / "temporal.json").read_text())
    if experiment not in report["experiments"] or Path(experiment).name != experiment:
        raise ValueError("Choose an existing single experiment label from temporal.json.")
    if report["source_interval_seconds"] != [2, 6] or report["target_interval_seconds"] != [6, 10]:
        raise ValueError("Unsupported source temporal intervals.")
    if report["cache_sha256"] != sha256(cache_path / "cache.json"):
        raise ValueError("Cache differs from temporal training.")
    verified_file(run, "settings.json", report["settings_sha256"])
    for name, digest in report["manifest_sha256"].items():
        verified_file(run, name, digest)
    training = pd.read_csv(run / "trial_splits.csv")
    if training.fold.isin([1, 2]).any():
        raise ValueError("Source temporal run contains reserved dyads.")
    paths = {"initial": verified_file(run, "initial_projection.pt", report["initial_projection_sha256"])}
    for arm in ("true", "shuffled"):
        paths[arm] = verified_file(
            run, f"{experiment}/{arm}/best_projection.pt", report["experiments"][experiment][arm]["projection_sha256"]
        )
    initialization = report["initialization"]
    if initialization == "pretrained" and sha256(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Backbone checkpoint differs from temporal training.")
    states = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in paths.items()}
    initial = states["initial"]
    for state in states.values():
        if any(not torch.isfinite(v).all() for v in state.values()) or (state["0.scale"] <= 0).any():
            raise ValueError("Invalid saved projection/statistics.")
        if not all(torch.equal(state[key], initial[key]) for key in ("0.mean", "0.scale")):
            raise ValueError("Selected head has different standardization from initial head.")
    scaler = SavedStandardizer(initial["0.mean"], initial["0.scale"])
    heads = {"pooled": nn.Identity(), "standardized_pooled": scaler}
    for name, state in states.items():
        head = nn.Sequential(
            copy.deepcopy(scaler), nn.Linear(state["1.weight"].shape[1], state["1.weight"].shape[0], bias=False)
        )
        head.load_state_dict(state, strict=True)
        head.requires_grad_(False)
        heads[name] = head
    config = SSLConfig(initialization=initialization, pairs=report["pairs"], seed=report["seed"])
    cache = EEGCache(cache_path, config)
    gallery_seed = report["seed"] + 2000
    rows, census = external_trials(cache, report["pairs"], gallery_seed)
    if set(rows.group) & set(training.group):
        raise ValueError("External and source-training dyads overlap.")
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output / "galleries.csv", index=False)
    census.to_csv(output / "census.csv", index=False)
    torch.manual_seed(report["seed"])
    model = ContrastiveLabram(initialization, checkpoint, False).to(device).eval()
    bank = temporal_features(model, cache, rows)["validation"]
    del model
    galleries = {key: np.arange(len(entry["a"])) for key, entry in bank.items()}
    temperature = (
        report.get("experiment_settings", {}).get(experiment, {}).get("temperature", float(experiment.split("_")[1]))
    )
    result = {
        "scope": "Exploratory unseen-dyad same-person temporal transfer; no training or external model selection",
        "source_run": str(run.resolve()),
        "source_report_sha256": sha256(run / "temporal.json"),
        "experiment": experiment,
        "temperature": temperature,
        "initialization": initialization,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "cache_sha256": report["cache_sha256"],
        "source_projection_sha256": {name: sha256(path) for name, path in paths.items()},
        "source_best_epochs": {
            arm: report["experiments"][experiment][arm]["best_epoch"] for arm in ("true", "shuffled")
        },
        "selection": "Source within-dyad best checkpoints, fixed before external evaluation",
        "standardization": "Saved training-only statistics; no refitting",
        "external_fold": 1,
        "test_evaluated": False,
        "training_performed": False,
        "gallery_seed": gallery_seed,
        "pairs": report["pairs"],
        "dyads": int(rows.group.nunique()),
        "paired_trials_evaluated": len(rows),
        "same_person_pairs_evaluated": 2 * len(rows),
        "source_interval_seconds": [2, 6],
        "target_interval_seconds": [6, 10],
        "manifest_sha256": {name: sha256(output / name) for name in ("galleries.csv", "census.csv")},
        "chance_top1": report["chance_top1"],
        "chance_mrr": report["chance_mrr"],
        "device": device,
        "torch_version": torch.__version__,
        "representations": {},
    }
    dyads = {}
    for name, head in heads.items():
        scores = temporal_metrics(head, bank, galleries, temperature)
        scores.to_csv(output / f"{name}_scores.csv", index=False)
        result["representations"][name] = temporal_summary(scores)
        dyads[name] = scores.groupby("group")[["mrr", "top1"]].mean()
    differences = pd.concat(
        {name: dyads["true"] - dyads[name] for name in ("pooled", "initial", "shuffled")}, names=["reference", "group"]
    ).reset_index()
    differences.to_csv(output / "paired_dyad_differences.csv", index=False)
    (output / "transfer.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_transfer(output, result, dyads)
    return result


def plot_transfer(output: Path, report: dict, dyads: dict) -> None:
    """Plot paired dyad trajectories and macro means for the locked representations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dyads)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ("mrr", "top1"), strict=True):
        for group in dyads["true"].index:
            ax.plot(names, [dyads[name].loc[group, metric] for name in names], alpha=0.4, marker="o")
        ax.plot(
            names,
            [report["representations"][name]["macro"][metric] for name in names],
            color="black",
            marker="o",
            linewidth=3,
            label="Dyad mean",
        )
        ax.axhline(report[f"chance_{metric}"], color="gray", linestyle="--", label="Chance")
        ax.set(ylabel=metric.upper(), ylim=(0, 1), title="Unseen-dyad temporal retrieval")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"transfer_comparison.{extension}", dpi=150)
    plt.close(fig)
