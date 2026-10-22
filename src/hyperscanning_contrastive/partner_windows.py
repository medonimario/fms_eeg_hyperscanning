"""Within-dyad partner learning from every usable exact-interval window set."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .contrastive import contrastive_loss
from .data import sha256
from .overfit import ContrastiveLabram
from .spectral import FeatureStandardizer
from .ssl import EEGCache, SSLConfig
from .temporal import temporal_metrics

SET_KEYS = ["split", "group", "condition", "crop_start_s"]
TRIAL_KEYS = ["group", "condition", "trial_id"]


def plan_windows(
    cache: EEGCache, validation_trials: int, seed: int, conditions: tuple[str, ...] = ("open",)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign whole paired trials before forming interval sets, including trials with no windows."""
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or not set(conditions) <= set(cache.metadata["conditions"])
    ):
        raise ValueError("Conditions must be unique, nonempty, and present in the cache.")
    if validation_trials < 2:
        raise ValueError("At least two validation trials are required.")
    if "trials.csv" not in cache.metadata["manifest_sha256"]:
        raise ValueError("All-window training requires a verified trials.csv, including fully excluded trials.")
    trials = pd.read_csv(cache.path / "trials.csv")
    rows = cache.rows.copy()
    if trials.duplicated(TRIAL_KEYS).any() or rows.duplicated(TRIAL_KEYS + ["crop_start_s"]).any():
        raise ValueError("Duplicate trial or interval identities.")
    if not np.isfinite(rows.crop_start_s).all() or (rows.crop_start_s < 0).any() or (rows.crop_start_s > 14).any():
        raise ValueError("Invalid window onset within the eighteen-second crop.")
    if not np.allclose(rows.crop_end_s - rows.crop_start_s, 4):
        raise ValueError("All windows must be four seconds long.")
    for _, frame in rows.groupby(TRIAL_KEYS):
        ordered = frame.sort_values("crop_start_s")
        if np.any(ordered.crop_start_s.to_numpy()[1:] < ordered.crop_end_s.to_numpy()[:-1]):
            raise ValueError("Overlapping windows in a trial.")
    for group, frame in trials.groupby("group"):
        if frame.fold.nunique() != 1 or cache.metadata["group_folds"].get(str(group)) != int(frame.fold.iloc[0]):
            raise ValueError("Trial fold differs from cache metadata.")
        if set(frame.condition) != set(cache.metadata["conditions"]):
            raise ValueError("Incomplete trial condition coverage.")
    trial_index = trials[TRIAL_KEYS + ["fold", "condition_trial"]]
    checked = rows.merge(trial_index, on=TRIAL_KEYS, how="left", validate="many_to_one", suffixes=("", "_trial"))
    if (
        checked.fold_trial.isna().any()
        or not (checked.fold == checked.fold_trial).all()
        or not (checked.condition_trial == checked.condition_trial_trial).all()
    ):
        raise ValueError("Window/trial manifest disagreement.")
    trials["split"] = "train"
    trials.loc[trials.fold == 1, "split"] = "external_validation"
    trials.loc[trials.fold == 2, "split"] = "test"
    rng = np.random.default_rng(seed)
    for _, frame in trials[trials.split == "train"].groupby(["group", "condition"], sort=True):
        frame = frame.sort_values("condition_trial")
        if len(frame) < validation_trials + 2:
            raise ValueError("Too few whole trials for the requested split.")
        trials.loc[rng.choice(frame.index.to_numpy(), validation_trials, replace=False), "split"] = "validation"
    trials["selected_condition"] = trials.condition.isin(conditions)
    rows = rows.merge(trials[TRIAL_KEYS + ["split", "selected_condition"]], on=TRIAL_KEYS, validate="many_to_one")
    census = rows.groupby(SET_KEYS).size().rename("candidates").reset_index()
    census["used"] = census.split.isin(["train", "validation"]) & (census.candidates >= 2)
    census["reason"] = np.where(
        ~census.split.isin(["train", "validation"]),
        "reserved_fold",
        np.where(census.candidates < 2, "fewer_than_two_trials", "used"),
    )
    census.loc[~census.condition.isin(conditions), "used"] = False
    census.loc[~census.condition.isin(conditions), "reason"] = "condition_not_selected"
    rows = rows.merge(census[SET_KEYS + ["used", "reason"]], on=SET_KEYS, validate="many_to_one")
    expected = set(
        map(
            tuple,
            trials[(trials.split == "train") & trials.selected_condition][["group", "condition"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    for split in ("train", "validation"):
        actual = set(
            map(tuple, rows[(rows.split == split) & rows.used][["group", "condition"]].drop_duplicates().to_numpy())
        )
        if not expected or actual != expected:
            raise ValueError(f"No usable interval set for some {split} dyad/conditions; see artifact coverage.")
    counts = rows.groupby(TRIAL_KEYS).size().rename("cached_windows")
    trials = trials.merge(counts, on=TRIAL_KEYS, how="left").fillna({"cached_windows": 0})
    return trials, rows, census


def extract_features(model: nn.Module, cache: EEGCache, rows: pd.DataFrame) -> dict:
    """Encode only usable development windows, keeping the two partner views distinct."""
    bank = {"train": {}, "validation": {}}
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for (split, group, condition, offset), frame in rows[rows.used].groupby(SET_KEYS, sort=True):
            frame = frame.sort_values("condition_trial").reset_index(drop=True)
            entry = {"rows": frame, "split": split}
            for member, name in enumerate(("a", "b")):
                windows = frame.assign(member=member)
                entry[name] = torch.cat(
                    [
                        model.features(cache.signals(windows.iloc[i : i + 8]).to(device)).cpu()
                        for i in range(0, len(windows), 8)
                    ]
                )
            bank[split][(int(group), condition, float(offset))] = entry
            print(f"Encoded {split} dyad {group}, {condition}, interval {offset:g}–{offset + 4:g}", flush=True)
    return bank


def score_windows(head: nn.Module, bank: dict, temperature: float, mappings: dict | None = None) -> pd.DataFrame:
    """Score all candidates per interval with exact chance values for varying gallery sizes."""
    galleries = {key: np.arange(len(entry["a"])) for key, entry in bank.items()}
    scores = temporal_metrics(head, bank, galleries, temperature, mappings).rename(columns={"member": "crop_start_s"})
    scores["candidates"] = [len(bank[(r.group, r.condition, r.crop_start_s)]["a"]) for r in scores.itertuples()]
    scores["queries"] = 2 * scores.candidates
    scores["chance_top1"] = 1 / scores.candidates
    scores["chance_mrr"] = [sum(1 / i for i in range(1, n + 1)) / n for n in scores.candidates]
    scores["chance_recall5"] = np.minimum(5, scores.candidates) / scores.candidates
    scores["chance_mean_rank"] = (scores.candidates + 1) / 2
    scores["mrr_excess"] = scores.mrr - scores.chance_mrr
    scores["top1_excess"] = scores.top1 - scores.chance_top1
    scores["onset_s"] = scores.crop_start_s + 2
    scores["end_s"] = scores.onset_s + 4
    return scores


def summarize(scores: pd.DataFrame) -> dict:
    """Weight intervals equally within condition, then conditions within dyad, then dyads."""
    values = scores.drop(columns=["crop_start_s", "onset_s", "end_s", "candidates", "queries"])
    grouped = values.groupby(["group", "condition"]).mean()
    return {
        "macro": grouped.groupby("group").mean().mean().to_dict(),
        "conditions": {str(name): frame.mean().to_dict() for name, frame in grouped.groupby(level="condition")},
    }


def validate_optimization(temperature: float, projection_lr: float, weight_decay: float) -> None:
    """Reject nonfinite or invalid optimization settings before reading EEG."""
    for name, value in (("temperature", temperature), ("projection_lr", projection_lr), ("weight_decay", weight_decay)):
        if not np.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
            raise ValueError(f"Invalid {name}: {value}")


def train_partner_windows(
    cache_path: Path,
    output: Path,
    checkpoint: Path,
    device: str = "cpu",
    initialization: str = "pretrained",
    validation_trials: int = 8,
    epochs: int = 20,
    seed: int = 42,
    split_seed: int = 2042,
    permutation_seed: int = 1042,
    groups_per_update: int = 4,
    conditions: tuple[str, ...] = ("open",),
    temperature: float = 0.05,
    projection_lr: float = 0.001,
    weight_decay: float = 0.01,
) -> dict:
    """Train fresh frozen-backbone heads on all usable intervals with matched shuffled controls."""
    validate_optimization(temperature, projection_lr, weight_decay)
    if groups_per_update < 1:
        raise ValueError("groups_per_update must be positive.")
    cache = EEGCache(cache_path, SSLConfig(initialization=initialization, pairs=2, epochs=epochs), all_windows=True)
    trials, rows, census = plan_windows(cache, validation_trials, split_seed, conditions)
    output.mkdir(parents=True, exist_ok=False)
    trials.to_csv(output / "trial_splits.csv", index=False)
    rows.to_csv(output / "windows.csv", index=False)
    census.to_csv(output / "census.csv", index=False)
    torch.manual_seed(seed)
    model = ContrastiveLabram(initialization, checkpoint, False).to(device).eval()
    bank = extract_features(model, cache, rows)
    del model
    return fit_window_bank(
        cache_path,
        output,
        checkpoint,
        bank,
        trials,
        rows,
        initialization,
        device,
        validation_trials,
        epochs,
        seed,
        split_seed,
        permutation_seed,
        groups_per_update,
        conditions,
        temperature=temperature,
        projection_lr=projection_lr,
        weight_decay=weight_decay,
    )


def fit_window_bank(
    cache_path: Path,
    output: Path,
    checkpoint: Path,
    bank: dict,
    trials: pd.DataFrame,
    rows: pd.DataFrame,
    initialization: str,
    device: str,
    validation_trials: int,
    epochs: int,
    seed: int,
    split_seed: int,
    permutation_seed: int,
    groups_per_update: int,
    conditions: tuple[str, ...],
    structured: bool = False,
    profile: str = "pooled",
    temperature: float = 0.05,
    projection_lr: float = 0.001,
    weight_decay: float = 0.01,
) -> dict:
    """Fit matched readouts on a prepared feature bank using the same galleries and schedule."""
    validate_optimization(temperature, projection_lr, weight_decay)
    scaler = FeatureStandardizer(bank["train"])
    torch.manual_seed(seed)
    if structured:
        positions, width = scaler.mean.shape
        initial = nn.Sequential(
            scaler, nn.Linear(width, 16, bias=False), nn.Flatten(1), nn.Linear(positions * 16, 64, bias=False)
        )
    else:
        initial = nn.Sequential(scaler, nn.Linear(scaler.mean.numel(), 64, bias=False))
    torch.save(initial.state_dict(), output / "initial_projection.pt")
    rng = np.random.default_rng(permutation_seed)
    mappings = {key: rng.permutation(len(entry["a"])) for key, entry in bank["train"].items()}
    mapping_rows = []
    for key, permutation in mappings.items():
        ids = bank["train"][key]["rows"].trial_id.tolist()
        mapping_rows.extend(
            {
                "group": key[0],
                "condition": key[1],
                "crop_start_s": key[2],
                "source_trial": trial,
                "target_trial": ids[permutation[i]],
            }
            for i, trial in enumerate(ids)
        )
    pd.DataFrame(mapping_rows).to_csv(output / "shuffled_training_pairs.csv", index=False)
    baselines = {}
    for name, head in (
        ("pooled", nn.Flatten(1)),
        ("standardized_pooled", nn.Sequential(scaler, nn.Flatten(1))),
        ("initial", initial),
    ):
        scores = score_windows(head, bank["validation"], temperature)
        scores.to_csv(output / f"validation_{name}.csv", index=False)
        baselines[name] = summarize(scores)
    report = {
        "scope": "Exploratory within-dyad partner retrieval using all usable exact-interval windows",
        "conditions": list(conditions),
        "condition_policy": "Only selected conditions enter standardization, training, controls, and selection",
        "trial_split_policy": "Split all cache conditions before filtering to preserve previous trial assignments",
        "excluded_condition_windows": int((~rows.selected_condition).sum()),
        "initialization": initialization,
        "seed": seed,
        "split_seed": split_seed,
        "permutation_seed": permutation_seed,
        "fixed_points_allowed": True,
        "validation_trials": validation_trials,
        "epochs": epochs,
        "groups_per_update": groups_per_update,
        "temperature": temperature,
        "projection_lr": projection_lr,
        "weight_decay": weight_decay,
        "gradient_clip": 1.0,
        "backbone_frozen": True,
        "readout_profile": profile,
        "projection": "shared 200-to-16 then position-specific linear-to-64" if structured else "linear-to-64",
        "trainable_parameters": sum(p.numel() for p in initial.parameters()),
        "feature_device": device,
        "projection_device": "cpu",
        "torch_version": torch.__version__,
        "external_validation_evaluated": False,
        "test_evaluated": False,
        "cache_sha256": sha256(cache_path / "cache.json"),
        "checkpoint_sha256": sha256(checkpoint) if initialization == "pretrained" else None,
        "initial_projection_sha256": sha256(output / "initial_projection.pt"),
        "manifest_sha256": {
            name: sha256(output / name)
            for name in ("trial_splits.csv", "windows.csv", "census.csv", "shuffled_training_pairs.csv")
        },
        "coverage": {
            split: {
                "dyads": int(frame.group.nunique()),
                "paired_trials": len(frame),
                "trials_with_no_cached_windows": int((frame.cached_windows == 0).sum()),
                "cached_paired_windows": int(((rows.split == split) & rows.selected_condition).sum()),
                "usable_paired_windows": int(((rows.split == split) & rows.used).sum()),
                "excluded_singleton_windows": int(
                    ((rows.split == split) & (rows.reason == "fewer_than_two_trials")).sum()
                ),
            }
            for split, frame in trials[trials.selected_condition].groupby("split")
        },
        "training_sets_per_epoch": len(bank["train"]),
        "updates_per_epoch": int(np.ceil(len(bank["train"]) / groups_per_update)),
        "training_paired_windows_per_epoch": sum(len(e["a"]) for e in bank["train"].values()),
        "aggregation": "equal intervals within dyad/condition, then equal conditions and dyads",
        "training_diagnostics": "full training galleries; differing candidate counts; compare chance excess",
        "selection": "maximum validation macro MRR, earliest tie; same fixed galleries at every epoch",
        "baselines": baselines,
    }
    (output / "settings.json").write_text(json.dumps(report, indent=2) + "\n")
    report["arms"] = {}
    keys = list(bank["train"])
    for arm in ("true", "shuffled"):
        folder = output / arm
        folder.mkdir()
        head = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(head.parameters(), lr=projection_lr, weight_decay=weight_decay)
        rng = np.random.default_rng(seed + 4000)
        best = -float("inf")
        history = []
        for epoch in range(epochs + 1):
            losses = []
            if epoch:
                order = rng.permutation(len(keys))
                for start in range(0, len(order), groups_per_update):
                    batch = order[start : start + groups_per_update]
                    optimizer.zero_grad(set_to_none=True)
                    for index in batch:
                        key = keys[index]
                        entry = bank["train"][key]
                        target = entry["b"][mappings[key]] if arm == "shuffled" else entry["b"]
                        loss = contrastive_loss(head(entry["a"]), head(target), temperature)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite partner loss.")
                        (loss / len(batch)).backward()
                        losses.append(loss.item())
                    nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
            validation = score_windows(head, bank["validation"], temperature)
            train_true = score_windows(head, bank["train"], temperature)
            objective = train_true if arm == "true" else score_windows(head, bank["train"], temperature, mappings)
            summary = summarize(validation)
            record = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, **summary["macro"]}
            for name, scores in (("train_true", train_true), ("train_objective", objective)):
                record.update({f"{name}_{k}": v for k, v in summarize(scores)["macro"].items()})
            history.append(record)
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            validation.to_csv(folder / f"validation_epoch_{epoch:03d}.csv", index=False)
            if record["mrr"] > best:
                best = record["mrr"]
                torch.save(head.state_dict(), folder / "best_projection.pt")
                validation.to_csv(folder / "validation_best.csv", index=False)
                metrics = validation.drop(columns=["crop_start_s", "onset_s", "end_s", "candidates", "queries"])
                per_condition = metrics.groupby(["group", "condition"]).mean()
                per_condition.to_csv(folder / "validation_best_by_dyad_condition.csv")
                per_condition.groupby("group").mean().to_csv(folder / "validation_best_by_dyad.csv")
                report["arms"][arm] = {"best_epoch": epoch, **summary}
            print(f"Partner windows {arm} epoch {epoch}: validation MRR={record['mrr']:.4f}", flush=True)
        torch.save(head.state_dict(), folder / "last_projection.pt")
        report["arms"][arm]["projection_sha256"] = sha256(folder / "best_projection.pt")
    report["settings_sha256"] = sha256(output / "settings.json")
    (output / "partner_windows.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_partner_windows(output, report)
    return report


def plot_partner_windows(output: Path, report: dict) -> None:
    """Plot validation ranks and chance-adjusted train/validation diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for arm, color in (("true", "#0072B2"), ("shuffled", "#D55E00")):
        history = pd.read_csv(output / arm / "history.csv")
        axes[0].plot(history.epoch, history.mrr, label=arm, color=color)
        axes[1].plot(history.epoch, history.top1, label=arm, color=color)
        axes[2].plot(history.epoch, history.mrr_excess, label=f"{arm}: validation", color=color)
        axes[2].plot(
            history.epoch,
            history.train_objective_mrr_excess,
            label=f"{arm}: train objective",
            color=color,
            linestyle="--",
        )
    for ax, metric in zip(axes[:2], ("mrr", "top1"), strict=True):
        ax.axhline(history[f"chance_{metric}"].iloc[0], color="gray", linestyle="--", label="Gallery-matched chance")
        ax.axhline(report["baselines"]["pooled"]["macro"][metric], color="black", linestyle=":", label="Raw readout")
        ax.set(ylabel=metric.upper(), ylim=(0, 1), title="Held-out whole trials")
    axes[2].axhline(0, color="gray", linestyle=":")
    axes[2].set(ylabel="MRR minus gallery-matched chance", title="Train/validation (different candidate counts)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Partner retrieval: " + ", ".join(report.get("conditions", ["all conditions"])))
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"partner_windows_curves.{extension}", dpi=150)
    plt.close(fig)
