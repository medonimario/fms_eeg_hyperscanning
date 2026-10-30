"""Training-free matching-patch retrieval from verified saved LaBraM features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CHANNELS, sha256
from .video_baseline import score_similarities


def patch_scores(a: np.ndarray, b: np.ndarray) -> tuple[dict, np.ndarray]:
    """Compare corresponding tokens, returning four scores and individual patch cosines."""
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (256, 200):
        raise ValueError("Expected matching K x 256 x 200 arrays.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite features.")
    an = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    patches = np.einsum("ipd,jpd->ijp", an, bn)
    af, bf = a.reshape(len(a), -1), b.reshape(len(b), -1)
    af = af / np.maximum(np.linalg.norm(af, axis=1, keepdims=True), 1e-12)
    bf = bf / np.maximum(np.linalg.norm(bf, axis=1, keepdims=True), 1e-12)
    return {
        "flattened": af @ bf.T,
        "patch_mean": patches.mean(-1),
        "patch_max": patches.max(-1),
        "patch_top8": np.partition(patches, -8, axis=-1)[..., -8:].mean(-1),
    }, patches


def winner_counts(patches: np.ndarray) -> dict:
    """Count maximizing patches, fractionally allocating patch and retrieval ties."""
    mask = np.isclose(patches, patches.max(-1, keepdims=True), atol=1e-10, rtol=0)
    weights = mask / mask.sum(-1, keepdims=True)
    k = len(patches)
    eye = np.eye(k, dtype=bool)
    result = {"aligned": weights[eye].sum(0), "mismatched": weights[~eye].sum(0)}
    result.update(retrieved_correct=np.zeros(256), retrieved_incorrect=np.zeros(256))
    for w, score in ((weights, patches.max(-1)), (weights.transpose(1, 0, 2), patches.max(-1).T)):
        for i in range(k):
            chosen = np.flatnonzero(np.isclose(score[i], score[i].max(), atol=1e-10, rtol=0))
            for j in chosen:
                label = "retrieved_correct" if i == j else "retrieved_incorrect"
                result[label] += w[i, j] / len(chosen)
    return result


def run_patch_scores(run: Path, output: Path) -> dict:
    """Reuse saved full tokens, training-only center and fixed galleries without EEG access."""
    report = json.loads((run / "projection.json").read_text())
    names = ("full_features.npy", "feature_windows.csv", "galleries.csv", "full_global/centers.npz")
    for name in names:
        expected = report["artifacts_sha256"].get(name)
        if expected is None or sha256(run / name) != expected:
            raise ValueError(f"Missing or mismatched source artifact: {name}")
    features = np.load(run / names[0], mmap_mode="r", allow_pickle=False)
    windows = pd.read_csv(run / names[1])
    galleries = pd.read_csv(run / names[2])
    if features.shape != (len(windows), 256, 200) or not np.isfinite(features).all():
        raise ValueError("Invalid full feature array.")
    if set(windows.split) != {"train", "validation"} or set(galleries.split) != {"train", "validation"}:
        raise ValueError("Only original train/validation splits are supported.")
    if windows.duplicated(["split", "participant_id", "cache_start"]).any():
        raise ValueError("Duplicate feature keys.")
    with np.load(run / names[3], allow_pickle=False) as saved:
        center = saved["global"]
    fitted = features[windows.split.to_numpy() == "train"].mean(0)
    if center.shape != (256, 200) or not np.allclose(center, fitted, atol=1e-5, rtol=1e-5):
        raise ValueError("Saved center does not match training-only feature mean.")
    keys = {(r.split, r.participant_id, int(r.cache_start)): i for i, r in enumerate(windows.itertuples())}
    rows, counts = [], {}
    for r in galleries.itertuples():
        starts = json.loads(r.starts)
        if len(starts) != 4 or len(set(starts)) != 4 or r.first == r.second:
            raise ValueError("Expected distinct people and four unique gallery moments.")
        a, b = [np.stack([features[keys[(r.split, p, int(t))]] - center for t in starts]) for p in (r.first, r.second)]
        scores, patches = patch_scores(a, b)
        for method, matrix in scores.items():
            rows.append(
                dict(split=r.split, method=method, first=r.first, second=r.second, **score_similarities(matrix))
            )
        for kind, values in winner_counts(patches).items():
            counts.setdefault((r.split, kind), np.zeros(256))
            counts[(r.split, kind)] += values
    frame = pd.DataFrame(rows)
    identities = pd.concat(
        [
            frame.rename(columns={"first": "participant_id"}).drop(columns="second"),
            frame.rename(columns={"second": "participant_id"}).drop(columns="first"),
        ]
    )
    people = identities.groupby(["split", "method", "participant_id"]).mean(numeric_only=True).reset_index()
    means = people.groupby(["split", "method"]).mean(numeric_only=True)
    for split in ("train", "validation"):
        expected = report["diagnostics"]["full"]["global"]["retrieval"][split]
        for metric in ("top1", "mrr"):
            if not np.isclose(means.loc[(split, "flattened"), metric], expected[metric], atol=1e-6):
                raise ValueError("Flattened scores do not reproduce source retrieval.")
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "pair_metrics.csv", index=False)
    people.to_csv(output / "person_metrics.csv", index=False)
    means.to_csv(output / "summary.csv")
    pd.DataFrame(
        [
            dict(
                split=s,
                kind=k,
                patch=p,
                channel=CHANNELS[p // 4],
                second=p % 4,
                count=float(v),
                fraction=float(v / values.sum()) if values.sum() else 0.0,
            )
            for (s, k), values in counts.items()
            for p, v in enumerate(values)
        ]
    ).to_csv(output / "winning_patches.csv", index=False)
    plot_scores(means, counts, output)
    result = dict(
        scope="Exploratory matching-patch scores; no training or significance test",
        source=str(run.resolve()),
        source_report_sha256=sha256(run / "projection.json"),
        source_artifacts_sha256={n: report["artifacts_sha256"][n] for n in names},
        centering="saved global training-only mean, verified by refitting",
        zero_norm_policy="epsilon 1e-12; zero-vector cosine is zero",
        aggregation="equal people after averaging bidirectional incident pair scores",
        winner_counts="descriptive gallery counts; ties fractional; not independent observations",
        flattened_reproduction_verified=True,
        chance_top1=0.25,
        chance_mrr=25 / 48,
        metrics={s: means.loc[s].to_dict(orient="index") for s in ("train", "validation")},
    )
    result["artifacts_sha256"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / "patch_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot_scores(means: pd.DataFrame, counts: dict, output: Path) -> None:
    """Export retrieval comparisons and descriptive maximum-patch frequencies."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, chance in zip(axes, ("top1", "mrr"), (0.25, 25 / 48), strict=True):
        for split in ("train", "validation"):
            ax.plot(means.loc[split].index, means.loc[split, metric], "o-", label=split)
        ax.axhline(chance, color="gray", linestyle=":")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle("Fixed galleries: global-centered matching-patch scoring")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"comparison.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    for ax, kind in zip(axes.flat, ("aligned", "mismatched", "retrieved_correct", "retrieved_incorrect"), strict=True):
        values = counts[("validation", kind)].reshape(64, 4).sum(1)
        ax.bar(range(64), values / max(values.sum(), 1))
        ax.set_xticks(range(64), CHANNELS, rotation=90, fontsize=6)
        ax.set_title(kind)
        ax.set_ylabel("Fraction of maximum-patch selections")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"winning_channels.{ext}", dpi=160)
    plt.close(fig)
