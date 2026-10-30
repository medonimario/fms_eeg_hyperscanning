"""Training-only CorrCA pilot for solo viewers of the no-noise movie."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from .data import CHANNELS, sha256


def fit_corrca(bank: dict[str, dict[int, np.ndarray]], components: int, shrinkage: float) -> tuple:
    """Fit shared filters using equally weighted moments and available-viewer covariance."""
    within, between = [], []
    for moment in sorted(set().union(*(set(v) for v in bank.values()))):
        values = [v[moment] for v in bank.values() if moment in v]
        if len(values) < 2:
            continue
        x = np.stack(values).astype(np.float64)
        x -= x.mean(axis=-1, keepdims=True)
        n, _, samples = x.shape
        diagonal = np.einsum("nct,ndt->cd", x, x, optimize=True)
        total = x.sum(axis=0)
        within.append(diagonal / (n * samples))
        between.append((total @ total.T - diagonal) / (n * (n - 1) * samples))
    if not within:
        raise ValueError("No training moments shared by at least two viewers.")
    rw, rb = np.mean(within, axis=0), np.mean(between, axis=0)
    regularized = (1 - shrinkage) * rw + shrinkage * np.trace(rw) / len(rw) * np.eye(len(rw))
    eigenvalues, filters = eigh(rb, regularized)
    return filters[:, -components:][:, ::-1], eigenvalues[::-1], rw


def waveform_features(x: np.ndarray, filters: np.ndarray | None) -> np.ndarray:
    """Normalize each waveform independently; no evaluation population statistics are fitted."""
    y = x.astype(np.float64) if filters is None else filters.T @ x
    y -= y.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(y, axis=-1, keepdims=True)
    return (y / np.maximum(norms, 1e-12)).reshape(-1) / np.sqrt(len(y))


def score_gallery(a: np.ndarray, b: np.ndarray) -> dict:
    """Score both directions, averaging metrics over random ordering of tied scores."""
    return score_similarities(a @ b.T)


def score_similarities(similarities: np.ndarray) -> dict:
    """Score a square similarity matrix with aligned diagonal and symmetric tie averaging."""
    ranks, top1, reciprocal = [], [], []
    for scores in (similarities, similarities.T):
        correct = np.diag(scores)[:, None]
        better = (scores > correct + 1e-10).sum(1)
        tied = (abs(scores - correct) <= 1e-10).sum(1)
        ranks.extend(1 + better + (tied - 1) / 2)
        top1.extend(np.where(better == 0, 1 / tied, 0))
        reciprocal.extend(np.mean(1 / np.arange(b + 1, b + n + 1)) for b, n in zip(better, tied, strict=True))
    ranks = np.array(ranks)
    return {
        "top1": float(np.mean(top1)),
        "mrr": float(np.mean(reciprocal)),
        "mean_rank": float(ranks.mean()),
        "aligned_correlation": float(np.diag(similarities).mean()),
        "mismatched_correlation": float(similarities[~np.eye(len(similarities), dtype=bool)].mean()),
    }


def run_video_baseline(
    cache: Path,
    output: Path,
    seed: int = 42,
    components: int = 3,
    shrinkage: float = 0.1,
    video_id: int = 0,
    noise_level: str = "none",
    reference: Path | None = None,
    cohort: str = "solo",
) -> dict:
    """Run a fixed K=4 held-time pilot without accessing reserved people's signals."""
    if not 0 < shrinkage <= 1 or not 1 <= components <= 64:
        raise ValueError("Require 0 < shrinkage <= 1 and 1..64 components.")
    if video_id not in (0, 3, 4, 5, 6, 7, 8, 9, 10) or noise_level not in ("none", "low", "high", "combined"):
        raise ValueError("Unknown video or noise level.")
    if (video_id == 0) != (noise_level == "none"):
        raise ValueError("Only video0 is no-noise.")
    if cohort not in ("solo", "together"):
        raise ValueError("Invalid cohort.")
    cache = cache.resolve()
    metadata = json.loads((cache / "cache.json").read_text())
    if (
        metadata["dataset"] != "shared-stimulus-video"
        or metadata["operation"] != "prepare"
        or metadata["channels"] != CHANNELS
        or metadata["sfreq"] != 200
        or metadata["units"] != "microvolts"
    ):
        raise ValueError("Expected prepared 200Hz video EEG cache in microvolts.")
    for name, digest in metadata["manifest_sha256"].items():
        if sha256(cache / name) != digest:
            raise ValueError(f"Manifest hash mismatch: {name}")
    windows = pd.read_csv(cache / "windows.csv")
    files = pd.read_csv(cache / "files.csv").set_index("array_path")
    people = pd.read_csv(cache / "participants.csv").fillna("")
    rng = np.random.default_rng(seed)
    if reference is not None:
        parent = json.loads((reference / "baseline.json").read_text())
        if parent["cache_sha256"] != sha256(cache / "cache.json"):
            raise ValueError("Participant split reference uses a different cache.")
        for name, digest in parent["artifacts_sha256"].items():
            if Path(name).name != name or sha256(reference / name) != digest:
                raise ValueError(f"Reference checksum mismatch: {name}")
        saved = pd.read_csv(reference / "participant_splits.csv").set_index("participant_id")
        if saved.index.has_duplicates or set(saved.index) != set(people.participant_id):
            raise ValueError("Reference participant identities differ.")
        people["split"] = people.participant_id.map(saved.split)
    else:
        people["split"] = "excluded"
        for _, stratum in people[people.exclusion_reason == ""].groupby(["condition", "counterbalance"]):
            groups = np.array(sorted(stratum.independence_group.unique()))
            reserved = set(rng.permutation(groups)[: max(1, int(np.ceil(len(groups) * 0.2)))])
            for idx, row in stratum.iterrows():
                people.loc[idx, "split"] = "reserved" if row.independence_group in reserved else "development"
    if (
        "independence_group" in people
        and people[people.split.isin(["development", "reserved"])].groupby("independence_group").split.nunique().max()
        > 1
    ):
        raise ValueError("Participant reservations split an independence group.")
    selected = windows[
        windows.eligible
        & (windows.condition == cohort)
        & (windows.video_id == video_id)
        & (
            windows.noise_level.isin(["low", "high"])
            if noise_level == "combined"
            else windows.noise_level == noise_level
        )
    ].copy()
    moments = sorted(selected.cache_start.unique())
    if len(moments) < 18:
        raise ValueError("Need at least 18 usable movie moments for this fixed pilot.")
    validation = set(moments[-8:])
    boundary = min(validation)
    selected["time_split"] = [
        "validation" if t in validation else "train" if t + 800 <= boundary - 800 else "guard"
        for t in selected.cache_start
    ]
    development = set(people.loc[people.split == "development", "participant_id"])
    selected["participant_split"] = selected.participant_id.map(people.set_index("participant_id").split)
    bank = {"train": {}, "validation": {}}
    for person, frame in selected[selected.participant_id.isin(development)].groupby("participant_id"):
        path = cache / frame.array_path.iloc[0]
        if not path.resolve().is_relative_to(cache):
            raise ValueError("Array path escapes cache.")
        if sha256(path) != files.loc[frame.array_path.iloc[0], "array_sha256"]:
            raise ValueError(f"Signal checksum mismatch: {person}")
        signal = np.load(path, mmap_mode="r", allow_pickle=False)
        if signal.ndim != 2 or signal.shape[0] != 64:
            raise ValueError("Invalid signal dimensions.")
        for split in bank:
            bank[split][person] = {}
            for row in frame[frame.time_split == split].itertuples():
                x = np.array(signal[:, row.cache_start : row.cache_stop])
                if x.shape != (64, 800) or not np.isfinite(x).all():
                    raise ValueError("Invalid cached window.")
                bank[split][person][row.cache_start] = x
    filters, eigenvalues, within = fit_corrca(bank["train"], components, shrinkage)
    shifted, shifts = {}, []
    for person, values in bank["train"].items():
        keys = sorted(values)
        if len(keys) < 4:
            raise ValueError(f"Insufficient training windows for {person}.")
        offset = int(rng.integers(2, len(keys) - 1))
        sources = np.roll(keys, offset)
        shifted[person] = {target: values[source] for target, source in zip(keys, sources, strict=True)}
        shifts.extend(
            {"participant_id": person, "target_start": int(t), "source_start": int(s)}
            for t, s in zip(keys, sources, strict=True)
        )
    control, _, _ = fit_corrca(shifted, components, shrinkage)
    output.mkdir(parents=True, exist_ok=False)
    people.to_csv(output / "participant_splits.csv", index=False)
    selected.to_csv(output / "window_splits.csv", index=False)
    pd.DataFrame(shifts).to_csv(output / "misaligned_training.csv", index=False)
    np.savez(output / "filters.npz", true=filters, misaligned=control, eigenvalues=eigenvalues, within=within)
    results, galleries, skipped = [], [], []
    for split, viewers in bank.items():
        for first, second in combinations(sorted(viewers), 2):
            common = sorted(set(viewers[first]) & set(viewers[second]))
            chosen = []
            for moment in common:
                if not chosen or moment - chosen[-1] >= 1600:
                    chosen.append(moment)
                if len(chosen) == 4:
                    break
            if len(chosen) < 4:
                skipped.append({"split": split, "first": first, "second": second})
                continue
            galleries.append({"split": split, "first": first, "second": second, "starts": json.dumps(chosen)})
            for name, projection in [("waveform", None), ("corrca", filters), ("misaligned_fit", control)]:
                features = [
                    np.stack([waveform_features(viewers[p][t], projection) for t in chosen]) for p in (first, second)
                ]
                results.append(
                    {"split": split, "model": name, "first": first, "second": second, **score_gallery(*features)}
                )
    scores = pd.DataFrame(results)
    if scores.empty or "validation" not in set(scores.split):
        raise ValueError("No valid validation galleries; inspect missing/QC windows.")
    scores.to_csv(output / "pair_metrics.csv", index=False)
    pd.DataFrame(galleries).to_csv(output / "galleries.csv", index=False)
    pd.DataFrame(skipped, columns=["split", "first", "second"]).to_csv(output / "skipped_pairs.csv", index=False)
    identities = pd.concat(
        [
            scores.rename(columns={"first": "participant_id"}).drop(columns="second"),
            scores.rename(columns={"second": "participant_id"}).drop(columns="first"),
        ]
    )
    metrics = list(score_gallery(np.eye(4), np.eye(4)))
    person_scores = identities.groupby(["split", "model", "participant_id"])[metrics].mean().reset_index()
    person_scores.to_csv(output / "person_metrics.csv", index=False)
    means = person_scores.groupby(["split", "model"])[metrics].mean()
    report = {
        "scope": f"Solo video{video_id} {noise_level}-noise held-time pilot; no significance test",
        "video_id": video_id,
        "noise_level": noise_level,
        "participant_reference_sha256": sha256(reference / "baseline.json") if reference else None,
        "cache_sha256": sha256(cache / "cache.json"),
        "seed": seed,
        "components": components,
        "shrinkage": shrinkage,
        "cohort": cohort,
        "pair_policy": "all eligible distinct viewers, including cross-dyad pairs",
        "sfreq": 200,
        "reserved_people_evaluated": False,
        "development_viewers": len(bank["train"]),
        "gallery_pair_counts": {s: sum(g["split"] == s for g in galleries) for s in bank},
        "development_independence_groups": int(
            people[people.participant_id.isin(bank["train"])].independence_group.nunique()
        )
        if "independence_group" in people
        else None,
        "training_person_windows": sum(map(len, bank["train"].values())),
        "validation_person_windows": sum(map(len, bank["validation"].values())),
        "validation_start_seconds": int(boundary) / 200,
        "boundary_guard_seconds": 4,
        "gallery_size": 4,
        "chance_top1": 0.25,
        "chance_mrr": sum(1 / np.arange(1, 5)) / 4,
        "aggregation": "equal people after averaging their available pair scores",
        "control": "training window-order circular shifts per person, at least two positions; no within-window wrap",
        "metrics": {split: means.loc[split].to_dict(orient="index") for split in means.index.levels[0]},
        "artifacts_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    }
    plot_baseline(person_scores, output, video_id)
    (output / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def plot_baseline(scores: pd.DataFrame, output: Path, video_id: int = 0) -> None:
    """Export retrieval and aligned-minus-mismatched correlation diagnostic figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = scores.copy()
    scores["correlation_margin"] = scores.aligned_correlation - scores.mismatched_correlation
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, chance in zip(axes, ["top1", "mrr", "correlation_margin"], [0.25, sum(1 / np.arange(1, 5)) / 4, 0]):
        for split, frame in scores.groupby("split"):
            means = frame.groupby("model")[metric].mean().reindex(["waveform", "corrca", "misaligned_fit"])
            ax.plot(means.index, means.values, marker="o", label=split)
        ax.axhline(chance, color="gray", linestyle="--")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend()
    fig.suptitle(f"Solo video{video_id}: descriptive pilot (model categories, not training epochs)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"baseline.{extension}", dpi=160)
    plt.close(fig)
