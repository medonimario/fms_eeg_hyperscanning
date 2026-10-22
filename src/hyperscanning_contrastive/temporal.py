"""Stage 2.5: non-overlapping within-person temporal retrieval on held-out whole trials."""

from __future__ import annotations

import copy
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .alignment_diagnostics import embedding_geometry
from .contrastive import contrastive_loss, pair_scores
from .data import sha256
from .overfit import ContrastiveLabram
from .partner import METRICS, query_metrics
from .spectral import FeatureStandardizer
from .ssl import EEGCache, SSLConfig


def temporal_trials(cache: EEGCache, pairs: int, split_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join clean offsets zero/four before splitting paired trials in original training dyads.

    No window is re-cut and no artifact exclusion is relaxed. Ineligible trial counts remain
    in the census of original training dyads. A fresh split is needed because both windows must be available.
    """
    rows = pd.read_csv(cache.path / "windows.csv")
    for offset in (0, 4):
        selected = rows[rows.crop_start_s == offset]
        if selected.duplicated(["group", "condition", "trial_id"]).any():
            raise ValueError("Duplicate trial windows at a requested temporal offset.")
    base = rows[rows.crop_start_s == 0].copy()
    targets = rows[rows.crop_start_s == 4][["group", "condition", "trial_id", "array", "array_index"]]
    targets = targets.rename(columns={"array": "target_array", "array_index": "target_array_index"})
    joined = base.merge(targets, on=["group", "condition", "trial_id"], how="inner", validate="one_to_one")
    joined = joined[~joined.fold.isin([1, 2])].copy()
    original = rows[~rows.fold.isin([1, 2])].groupby(["group", "condition"]).trial_id.nunique()
    eligible = joined.groupby(["group", "condition"]).size()
    census = original.rename("cached_trials").to_frame().join(eligible.rename("eligible_trials")).fillna(0)
    census["eligible_trials"] = census.eligible_trials.astype(int)
    census["excluded_trials"] = census.cached_trials - census.eligible_trials
    if (census.eligible_trials < 2 * pairs).any():
        bad = census[census.eligible_trials < 2 * pairs]
        raise ValueError(
            f"Require at least {2 * pairs} clean temporal pairs per dyad/condition; deficient strata:\n{bad}"
        )
    joined["split"] = "train"
    rng = np.random.default_rng(split_seed)
    for _, frame in joined.groupby(["group", "condition"], sort=True):
        frame = frame.sort_values("condition_trial")
        joined.loc[rng.choice(frame.index.to_numpy(), pairs, replace=False), "split"] = "validation"
    return joined.sort_values(["group", "condition", "condition_trial"]).reset_index(drop=True), census.reset_index()


def temporal_features(model: nn.Module, cache: EEGCache, rows: pd.DataFrame) -> dict:
    """Cache source and target features separately for each split/person/condition."""
    bank = {"train": {}, "validation": {}}
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for (split, group, condition), frame in rows.groupby(["split", "group", "condition"], sort=True):
            for member in (0, 1):
                entry = {"rows": frame.copy(), "split": split}
                for label in ("a", "b"):
                    windows = frame.assign(member=member)
                    if label == "b":
                        windows = windows.assign(array=frame.target_array, array_index=frame.target_array_index)
                    chunks = [
                        model.features(cache.signals(windows.iloc[i : i + 8]).to(device)).cpu()
                        for i in range(0, len(windows), 8)
                    ]
                    entry[label] = torch.cat(chunks)
                bank[split][(int(group), condition, member)] = entry
            print(f"Extracted temporal features: {split} dyad {group}, {condition}", flush=True)
    return bank


def temporal_metrics(
    head: nn.Module, bank: dict, galleries: dict, temperature: float, mappings: dict | None = None
) -> pd.DataFrame:
    """Score fixed galleries with true or assigned temporal correspondence, both directions."""
    records = []
    head.eval()
    with torch.no_grad():
        for key, indices in galleries.items():
            entry = bank[key]
            target = mappings[key][indices] if mappings is not None else indices
            a, b = head(entry["a"][indices]), head(entry["b"][target])
            scores = pair_scores(a, b)
            values = query_metrics(scores)[METRICS].mean().to_dict()
            positive = scores.diagonal().mean().item()
            negative = scores[~torch.eye(len(scores), dtype=torch.bool)].mean().item()
            geometry_a, geometry_b = embedding_geometry(a), embedding_geometry(b)
            records.append(
                {
                    "group": key[0],
                    "condition": key[1],
                    "member": key[2],
                    **values,
                    "loss": contrastive_loss(a, b, temperature).item(),
                    "cosine_margin": positive - negative,
                    **{k: (v + geometry_b[k]) / 2 for k, v in geometry_a.items()},
                }
            )
    return pd.DataFrame(records)


def temporal_summary(frame: pd.DataFrame) -> dict:
    """Average persons then conditions then dyads, retaining condition-level results."""
    values = frame.drop(columns="member").groupby(["group", "condition"]).mean()
    return {
        "macro": values.groupby("group").mean().mean().to_dict(),
        "conditions": {
            condition: part.droplevel("condition").mean().to_dict()
            for condition, part in values.groupby(level="condition")
        },
    }


def run_temporal(
    cache_path: Path,
    output: Path,
    checkpoint: Path,
    device: str = "cpu",
    initialization: str = "pretrained",
    pairs: int = 8,
    epochs: int = 20,
    seed: int = 42,
    split_seed: int = 2042,
    permutation_seed: int = 1042,
    temperatures: tuple[float, ...] = (0.05, 0.1, 0.2),
    groups_per_update: int = 4,
    learning_rates: tuple[float, ...] = (0.001,),
    weight_decays: tuple[float, ...] = (0.01,),
) -> dict:
    """Run one frozen feature extraction and a bounded, matched temperature experiment.

    Training uses all remaining trials within each person/condition as negatives and visits
    every stratum once per epoch. Multiple stratum losses are averaged per optimizer step;
    candidates never cross participant or condition boundaries. Validation uses fixed K trials.
    """
    if (
        not temperatures
        or len(set(temperatures)) != len(temperatures)
        or any(not np.isfinite(t) or t <= 0 for t in temperatures)
        or groups_per_update < 1
    ):
        raise ValueError("Require distinct positive finite temperatures and a positive group batch size.")
    for name, values, allow_zero in (("learning rates", learning_rates, False), ("weight decays", weight_decays, True)):
        if (
            not values
            or len(set(values)) != len(values)
            or any(not np.isfinite(v) or v < 0 or (v == 0 and not allow_zero) for v in values)
        ):
            raise ValueError(
                f"Invalid {name}: require distinct finite {'nonnegative' if allow_zero else 'positive'} values."
            )
    config = SSLConfig(initialization=initialization, pairs=pairs, epochs=epochs, seed=seed)
    cache = EEGCache(cache_path, config)
    rows, census = temporal_trials(cache, pairs, split_seed)
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output / "trial_splits.csv", index=False)
    census.to_csv(output / "census.csv", index=False)
    torch.manual_seed(seed)
    model = ContrastiveLabram(initialization, checkpoint, False).to(device).eval()
    bank = temporal_features(model, cache, rows)
    del model
    scaler = FeatureStandardizer(bank["train"])
    width = scaler.mean.numel()
    torch.manual_seed(seed)
    initial_head = nn.Sequential(copy.deepcopy(scaler), nn.Linear(width, 64, bias=False))
    torch.save(initial_head.state_dict(), output / "initial_projection.pt")
    rng = np.random.default_rng(permutation_seed)
    mappings = {key: rng.permutation(len(entry["a"])) for key, entry in bank["train"].items()}
    mapping_rows = []
    for key, permutation in mappings.items():
        trials = bank["train"][key]["rows"].trial_id.tolist()
        mapping_rows.extend(
            {
                "group": key[0],
                "condition": key[1],
                "member": key[2],
                "source_trial": trial,
                "target_trial": trials[permutation[i]],
            }
            for i, trial in enumerate(trials)
        )
    pd.DataFrame(mapping_rows).to_csv(output / "shuffled_training_pairs.csv", index=False)
    rng = np.random.default_rng(seed + 1000)
    galleries = {
        split: {key: rng.choice(len(entry["a"]), pairs, replace=False) for key, entry in entries.items()}
        for split, entries in bank.items()
    }
    gallery_rows = []
    for split, selections in galleries.items():
        for key, indices in selections.items():
            gallery_rows.append(bank[split][key]["rows"].iloc[indices].assign(member=key[2]))
    pd.concat(gallery_rows).to_csv(output / "galleries.csv", index=False)
    baselines = {}
    for name, head in (
        ("pooled", nn.Identity()),
        ("standardized_pooled", scaler),
        ("initial_projection", initial_head),
    ):
        metrics = temporal_metrics(head, bank["validation"], galleries["validation"], 0.1)
        metrics.to_csv(output / f"validation_{name}.csv", index=False)
        baselines[name] = temporal_summary(metrics)
    sizes = [len(entry["a"]) for entry in bank["train"].values()]
    report = {
        "scope": "Stage 2.5 exploratory within-person non-overlapping temporal retrieval; not hyperscanning",
        "initialization": initialization,
        "seed": seed,
        "split_seed": split_seed,
        "permutation_seed": permutation_seed,
        "temperatures": list(temperatures),
        "epochs": epochs,
        "pairs": pairs,
        "groups_per_update": groups_per_update,
        "training_pairs_per_gallery_range": [min(sizes), max(sizes)],
        "negatives_per_training_query_range": [min(sizes) - 1, max(sizes) - 1],
        "pair_presentations_per_epoch": sum(sizes),
        "updates_per_epoch": int(np.ceil(len(sizes) / groups_per_update)),
        "epoch_definition": "one pass over every training trial for each participant",
        "source_interval_seconds": [2, 6],
        "target_interval_seconds": [6, 10],
        "split_counts": {
            s: {"dyads": int(f.group.nunique()), "paired_trials": len(f)} for s, f in rows.groupby("split")
        },
        "split_note": "new split after requiring both windows; distinct from prior partner galleries",
        "external_validation_evaluated": False,
        "test_evaluated": False,
        "feature_device": device,
        "projection_device": "cpu",
        "backbone_frozen": True,
        "projection": f"training-only standardization then bias-free {width}-to-64 linear",
        "projection_lr": learning_rates[0] if len(learning_rates) == 1 else None,
        "learning_rates": list(learning_rates),
        "weight_decay": weight_decays[0] if len(weight_decays) == 1 else None,
        "weight_decays": list(weight_decays),
        "gradient_clip": 1.0,
        "checkpoint_sha256": sha256(checkpoint) if initialization == "pretrained" else None,
        "cache_sha256": sha256(cache_path / "cache.json"),
        "manifest_sha256": {
            name: sha256(output / name)
            for name in ("trial_splits.csv", "galleries.csv", "shuffled_training_pairs.csv", "census.csv")
        },
        "initial_projection_sha256": sha256(output / "initial_projection.pt"),
        "torch_version": torch.__version__,
        "chance_top1": 1 / pairs,
        "chance_mrr": sum(1 / i for i in range(1, pairs + 1)) / pairs,
        "baselines": baselines,
        "experiments": {},
    }
    (output / "settings.json").write_text(json.dumps(report, indent=2) + "\n")
    keys = list(bank["train"])
    report["experiment_settings"] = {}
    for temperature, learning_rate, weight_decay in product(temperatures, learning_rates, weight_decays):
        label = f"temperature_{temperature:g}"
        if learning_rates != (0.001,) or weight_decays != (0.01,):
            label += f"_lr_{learning_rate:g}_wd_{weight_decay:g}"
        report["experiment_settings"][label] = {
            "temperature": temperature,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        report["experiments"][label] = {}
        for arm in ("true", "shuffled"):
            folder = output / label / arm
            folder.mkdir(parents=True)
            head = copy.deepcopy(initial_head)
            optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
            rng = np.random.default_rng(seed + 4000)
            best, best_epoch = baselines["initial_projection"]["macro"]["mrr"], 0
            best_rows = temporal_metrics(head, bank["validation"], galleries["validation"], temperature)
            torch.save(head.state_dict(), folder / "best_projection.pt")
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
                                raise ValueError("Nonfinite temporal contrastive loss.")
                            (loss / len(batch)).backward()
                            losses.append(loss.item())
                        nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                validation = temporal_metrics(head, bank["validation"], galleries["validation"], temperature)
                train_true = temporal_metrics(head, bank["train"], galleries["train"], temperature)
                objective = (
                    train_true
                    if arm == "true"
                    else temporal_metrics(head, bank["train"], galleries["train"], temperature, mappings)
                )
                record = {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)) if losses else None,
                    **temporal_summary(validation)["macro"],
                }
                for name, metrics in (
                    ("validation", validation),
                    ("train_true", train_true),
                    ("train_objective", objective),
                ):
                    record.update({f"{name}_{k}": v for k, v in temporal_summary(metrics)["macro"].items()})
                history.append(record)
                pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
                validation.to_csv(folder / f"validation_epoch_{epoch:03d}.csv", index=False)
                if record["mrr"] > best:
                    best, best_epoch, best_rows = record["mrr"], epoch, validation.copy()
                    torch.save(head.state_dict(), folder / "best_projection.pt")
                print(f"{label} {arm} epoch {epoch}: validation MRR={record['mrr']:.4f}", flush=True)
            best_rows.to_csv(folder / "validation_best.csv", index=False)
            torch.save(head.state_dict(), folder / "last_projection.pt")
            report["experiments"][label][arm] = {
                "best_epoch": best_epoch,
                **temporal_summary(best_rows),
                "projection_sha256": sha256(folder / "best_projection.pt"),
            }
    report["selected_experiment"] = max(
        report["experiments"], key=lambda k: report["experiments"][k]["true"]["macro"]["mrr"]
    )
    report["selected_temperature"] = (
        f"temperature_{report['experiment_settings'][report['selected_experiment']]['temperature']:g}"
    )
    report["selected_settings"] = report["experiment_settings"][report["selected_experiment"]]
    report["selection_note"] = (
        "Selected by true-arm validation MRR only; same-configuration shuffled control; exploratory peak"
    )
    report["settings_sha256"] = sha256(output / "settings.json")
    (output / "temporal.json").write_text(json.dumps(report, indent=2) + "\n")
    comparison = []
    for label, arms in report["experiments"].items():
        for arm, result in arms.items():
            comparison.append(
                {
                    "experiment": label,
                    "arm": arm,
                    **report["experiment_settings"][label],
                    "best_epoch": result["best_epoch"],
                    **result["macro"],
                }
            )
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    plot_temporal(output)
    return report


def plot_temporal(run: Path) -> None:
    """Plot all temperatures, train/validation retrieval, and initial/chance baselines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((run / "temporal.json").read_text())
    labels = list(report["experiments"])
    fig, axes = plt.subplots(2, len(labels), figsize=(5 * len(labels), 8), squeeze=False)
    for column, label in enumerate(labels):
        for arm, color in (("true", "#0072B2"), ("shuffled", "#D55E00")):
            history = pd.read_csv(run / label / arm / "history.csv")
            for row, metric in enumerate(("mrr", "train_objective_mrr")):
                axes[row, column].plot(history.epoch, history[metric], label=arm, color=color)
        for row in range(2):
            ax = axes[row, column]
            ax.axhline(report["chance_mrr"], color="gray", linestyle="--", label="chance")
            title = label.replace("_lr_", "\nlr=").replace("_wd_", " wd=")
            ax.set(
                xlabel="Epoch",
                ylabel="MRR",
                title=f"{title}\n{'validation' if row == 0 else 'training objective'}",
            )
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
        for name, color in (("pooled", "#777777"), ("initial_projection", "#009E73")):
            axes[0, column].axhline(report["baselines"][name]["macro"]["mrr"], color=color, linestyle=":", label=name)
        axes[0, column].legend(fontsize=8)
    fig.suptitle("Same person, different non-overlapping windows: held-out whole trials", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(run / f"temporal_curves.{suffix}", dpi=160)
    plt.close(fig)
