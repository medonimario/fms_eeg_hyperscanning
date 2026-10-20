"""Frozen pooled-feature partner retrieval on development validation dyads."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .contrastive import pair_scores
from .data import sha256
from .overfit import ContrastiveLabram
from .ssl import EEGCache, SSLConfig

METRICS = ["top1", "recall5", "mrr", "mean_rank"]


def query_metrics(scores: torch.Tensor) -> pd.DataFrame:
    """Score diagonal matches per query in both directions with exact random-tie expectations."""
    positive = scores.diagonal()
    harmonic = torch.cat([scores.new_zeros(1), (1 / torch.arange(1, len(scores) + 1)).cumsum(0)])
    rows = []
    for direction, matrix in (("a_to_b", scores), ("b_to_a", scores.T)):
        greater = (matrix > positive[:, None]).sum(1)
        equal = (matrix == positive[:, None]).sum(1)
        frame = pd.DataFrame(
            {
                "direction": direction,
                "query_index": np.arange(len(scores)),
                "top1": ((greater == 0).float() / equal).numpy(),
                "recall5": ((min(5, len(scores)) - greater).clamp(min=0).minimum(equal) / equal).numpy(),
                "mrr": ((harmonic[greater + equal] - harmonic[greater]) / equal).numpy(),
                "mean_rank": (1 + greater + (equal - 1) / 2).numpy(),
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def partner_baseline(
    cache_path: Path,
    output: Path,
    checkpoint: Path,
    initialization: str = "pretrained",
    device: str = "cpu",
    pairs: int = 8,
    seed: int = 42,
    permutations: int = 200,
    permutation_seed: int = 1042,
) -> dict:
    """Evaluate unaugmented aligned partners, with no adaptation and no test-set scores.

    Uniform correspondence shuffles are exploratory controls. They do not adjust for trial order, drift,
    common cues, or behavioral confounds, and no physiological significance claim is made.
    """
    if permutations < 20:
        raise ValueError("Require at least 20 permutations.")
    config = SSLConfig(initialization=initialization, pairs=pairs, seed=seed)
    cache = EEGCache(cache_path, config)
    galleries = [frame for frame in cache.galleries("validation", config) if int(frame.member.iloc[0]) == 0]
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(galleries).to_csv(output / "galleries.csv", index=False)
    torch.manual_seed(seed)
    model = ContrastiveLabram(initialization, checkpoint, False).to(device).eval()
    rng = np.random.default_rng(permutation_seed)
    actual, null, matrices = [], [], {}
    with torch.no_grad():
        for frame in galleries:
            group, condition = int(frame.group.iloc[0]), frame.condition.iloc[0]
            a = model.features(cache.signals(frame.assign(member=0)).to(device)).cpu()
            b = model.features(cache.signals(frame.assign(member=1)).to(device)).cpu()
            scores = pair_scores(a, b)
            matrices[f"{group}_{condition}"] = scores.numpy()
            rows = query_metrics(scores)
            rows["trial_id"] = [frame.trial_id.iloc[i] for i in rows.query_index]
            actual.append(rows.assign(group=group, condition=condition))
            for repeat in range(permutations):
                permuted = scores[:, rng.permutation(pairs)]
                metrics = query_metrics(permuted)[METRICS].mean().to_dict()
                null.append({"group": group, "condition": condition, "repeat": repeat, **metrics})
            print(f"Dyad {group}, {condition}: top1={rows.top1.mean():.3f}, MRR={rows.mrr.mean():.3f}", flush=True)
    actual_frame, null_frame = pd.concat(actual, ignore_index=True), pd.DataFrame(null)
    actual_frame.to_csv(output / "queries.csv", index=False)
    per_dyad = actual_frame.groupby(["group", "condition", "direction"])[METRICS].mean()
    per_dyad.to_csv(output / "directions.csv")
    per_dyad = per_dyad.groupby(["group", "condition"]).mean()
    per_dyad.to_csv(output / "dyads.csv")
    null_frame.to_csv(output / "shuffled_dyads.csv", index=False)
    np.savez_compressed(output / "similarities.npz", **matrices)
    conditions = {}
    for condition in sorted(per_dyad.index.get_level_values("condition").unique()):
        observed = per_dyad.xs(condition, level="condition")
        shuffled = null_frame[null_frame.condition == condition].groupby("repeat")[METRICS].mean()
        conditions[condition] = {
            "dyads": len(observed),
            "actual": observed.mean().to_dict(),
            "dyad_sd": observed.std().to_dict(),
            "shuffled_mean": shuffled.mean().to_dict(),
            "shuffled_mean_mc_se": (shuffled.std() / np.sqrt(permutations)).to_dict(),
        }
    contrast = {}
    if {"open", "closed1", "closed2"} <= set(conditions):
        differences = (
            per_dyad.xs("open", level="condition")
            - (per_dyad.xs("closed1", level="condition") + per_dyad.xs("closed2", level="condition")) / 2
        )
        differences.to_csv(output / "open_minus_closed_by_dyad.csv")
        contrast = {"mean": differences.mean().to_dict(), "dyad_sd": differences.std().to_dict()}
    report = {
        "scope": "Exploratory validation-only frozen partner retrieval; no training or significance test",
        "initialization": initialization,
        "features": "original pooled LaBraM; no projection or augmentation",
        "checkpoint_sha256": sha256(checkpoint) if initialization == "pretrained" else None,
        "cache_sha256": sha256(cache_path / "cache.json"),
        "gallery_sha256": sha256(output / "galleries.csv"),
        "seed": seed,
        "permutation_seed": permutation_seed,
        "permutations": permutations,
        "fixed_points_allowed": True,
        "pairs": pairs,
        "validation_fold": 1,
        "test_fold": 2,
        "test_evaluated": False,
        "crop_offset_seconds": 0,
        "source_epoch_interval_seconds": [2, 6],
        "chance_top1": 1 / pairs,
        "chance_mrr": sum(1 / i for i in range(1, pairs + 1)) / pairs,
        "device": device,
        "torch_version": torch.__version__,
        "conditions": conditions,
        "open_minus_mean_closed": contrast,
    }
    (output / "partner.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
