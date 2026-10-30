"""Frozen pooled LaBraM retrieval on exactly the saved video pilot galleries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import CHANNELS, sha256
from .overfit import ContrastiveLabram
from .video_baseline import score_gallery


def load_reference(cache: Path, run: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Verify provenance and gallery membership before reading any EEG arrays."""
    report = json.loads((run / "baseline.json").read_text())
    metadata = json.loads((cache / "cache.json").read_text())
    expected = dict(
        dataset="shared-stimulus-video",
        operation="prepare",
        channels=CHANNELS,
        sfreq=200,
        units="microvolts",
        model_input_divisor=100,
        window_seconds=4,
    )
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Incompatible video cache.")
    if sha256(cache / "cache.json") != report["cache_sha256"]:
        raise ValueError("Reference used a different cache.")
    for root, hashes in [(cache, metadata["manifest_sha256"]), (run, report["artifacts_sha256"])]:
        for name, digest in hashes.items():
            if Path(name).name != name or sha256(root / name) != digest:
                raise ValueError(f"Artifact checksum mismatch: {name}")
    people = pd.read_csv(run / "participant_splits.csv").set_index("participant_id")
    if (
        "independence_group" in people
        and people[people.split.isin(["development", "reserved"])].groupby("independence_group").split.nunique().max()
        > 1
    ):
        raise ValueError("Participant reservations split an independence group.")
    windows = pd.read_csv(run / "window_splits.csv")
    source = pd.read_csv(cache / "windows.csv")
    cohort = report.get("cohort", "solo")
    video_id = int(report.get("video_id", 0))
    source = source[source.video_id == video_id].set_index(["participant_id", "cache_start"])
    if source.index.has_duplicates or windows.duplicated(["participant_id", "cache_start"]).any():
        raise ValueError("Duplicate window identity.")
    windows = windows.set_index(["participant_id", "cache_start"])
    galleries = pd.read_csv(run / "galleries.csv")
    if galleries.empty or set(galleries.split) != {"train", "validation"}:
        raise ValueError("Expected nonempty training and validation galleries.")
    for row in galleries.itertuples():
        starts = json.loads(row.starts)
        if (
            len(starts) != 4
            or any(type(t) is not int for t in starts)
            or any(b - a < 1600 for a, b in zip(starts, starts[1:]))
            or row.first == row.second
        ):
            raise ValueError("Invalid four-moment gallery.")
        for person in (row.first, row.second):
            if people.loc[person, "split"] != "development" or people.loc[person, "condition"] != cohort:
                raise ValueError("Gallery accesses a reserved or wrong-cohort participant.")
            for start in starts:
                saved, original = windows.loc[(person, start)], source.loc[(person, start)]
                if (
                    saved.time_split != row.split
                    or not original.eligible
                    or original.video_id != video_id
                    or original.condition != cohort
                    or (
                        report.get("noise_level", "combined") != "combined"
                        and original.noise_level != report["noise_level"]
                    )
                    or original.cache_stop != start + 800
                    or saved.array_path != original.array_path
                    or saved.cache_stop != original.cache_stop
                ):
                    raise ValueError("Gallery window disagrees with source or split manifest.")
    train = windows[windows.time_split == "train"]
    val = windows[windows.time_split == "validation"]
    if train.cache_stop.max() + 800 > val.index.get_level_values("cache_start").min():
        raise ValueError("Insufficient temporal split guard.")
    return report, galleries, windows


def run_video_labram(
    cache: Path,
    run: Path,
    output: Path,
    checkpoint: Path,
    initialization: str = "pretrained",
    device: str = "cpu",
    batch_size: int = 16,
    seed: int = 42,
) -> dict:
    """Extract unaugmented frozen features once per used window and compare fixed galleries."""
    if batch_size < 1 or initialization not in ("pretrained", "random"):
        raise ValueError("Invalid batch size or initialization.")
    cache, run = cache.resolve(), run.resolve()
    reference, galleries, windows = load_reference(cache, run)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = ContrastiveLabram(initialization, checkpoint, False).to(device).eval()
    model.requires_grad_(False)
    keys = sorted(
        {
            (person, start)
            for row in galleries.itertuples()
            for person in (row.first, row.second)
            for start in json.loads(row.starts)
        }
    )
    files = pd.read_csv(cache / "files.csv").dropna(subset=["array_path"]).set_index("array_path")
    features, current_path, array = [], None, None
    with torch.inference_mode():
        for offset in range(0, len(keys), batch_size):
            signals = []
            for key in keys[offset : offset + batch_size]:
                row = windows.loc[key]
                path = (cache / row.array_path).resolve()
                if not path.is_relative_to(cache):
                    raise ValueError("Signal path escapes cache.")
                if path != current_path:
                    if sha256(path) != files.loc[row.array_path, "array_sha256"]:
                        raise ValueError("Signal checksum mismatch.")
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    if array.ndim != 2 or array.shape[0] != 64 or array.dtype != np.float32:
                        raise ValueError("Expected float32 64-channel continuous EEG.")
                    current_path = path
                x = np.array(array[:, key[1] : row.cache_stop])
                if x.shape != (64, 800) or not np.isfinite(x).all():
                    raise ValueError("Invalid EEG window.")
                signals.append(x)
            pooled = model.features(torch.from_numpy(np.stack(signals)).to(device)).cpu().numpy()
            if pooled.ndim != 2 or len(pooled) != len(signals) or not np.isfinite(pooled).all():
                raise ValueError("Invalid pooled LaBraM features.")
            features.extend(pooled)
            print(f"Encoded {min(offset + batch_size, len(keys))}/{len(keys)} windows", flush=True)
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero-norm LaBraM embedding.")
    normalized = dict(zip(keys, features / norms, strict=True))
    label = f"labram_{initialization}"
    results = []
    for row in galleries.itertuples():
        a, b = [np.stack([normalized[(p, t)] for t in json.loads(row.starts)]) for p in (row.first, row.second)]
        result = score_gallery(a, b)
        result["aligned_cosine"] = result.pop("aligned_correlation")
        result["mismatched_cosine"] = result.pop("mismatched_correlation")
        results.append(dict(split=row.split, model=label, first=row.first, second=row.second, **result))
    pairs = pd.DataFrame(results)
    identities = pd.concat(
        [
            pairs.rename(columns={"first": "participant_id"}).drop(columns="second"),
            pairs.rename(columns={"second": "participant_id"}).drop(columns="first"),
        ]
    )
    metrics = ["top1", "mrr", "mean_rank", "aligned_cosine", "mismatched_cosine"]
    persons = identities.groupby(["split", "model", "participant_id"])[metrics].mean().reset_index()
    baseline = pd.read_csv(run / "person_metrics.csv")
    comparison = pd.concat(
        [
            baseline[["split", "model", "participant_id", "top1", "mrr", "mean_rank"]],
            persons[["split", "model", "participant_id", "top1", "mrr", "mean_rank"]],
        ]
    )
    pairs.to_csv(output / "pair_metrics.csv", index=False)
    persons.to_csv(output / "person_metrics.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(keys, columns=["participant_id", "cache_start"]).to_csv(output / "feature_windows.csv", index=False)
    np.save(output / "pooled_features.npy", features, allow_pickle=False)
    for name in ("galleries.csv", "participant_splits.csv", "window_splits.csv"):
        shutil.copyfile(run / name, output / name)
    plot_comparison(comparison, output)
    report = dict(
        scope="Frozen pooled LaBraM, same video0 pilot galleries; no training or significance test",
        reference_sha256=sha256(run / "baseline.json"),
        cache_sha256=sha256(cache / "cache.json"),
        gallery_sha256=sha256(output / "galleries.csv"),
        initialization=initialization,
        checkpoint_sha256=sha256(checkpoint) if initialization == "pretrained" else None,
        seed=seed,
        device=device,
        torch_version=torch.__version__,
        batch_size=batch_size,
        extracted_windows=len(keys),
        feature_width=features.shape[1],
        reserved_people_evaluated=False,
        representation="Original pooled LaBraM, cosine similarity; no projection, centering or augmentation",
        input_divisor=100,
        chance_top1=reference["chance_top1"],
        chance_mrr=reference["chance_mrr"],
        metrics={
            split: frame.groupby("model")[["top1", "mrr", "mean_rank"]].mean().to_dict(orient="index")
            for split, frame in comparison.groupby("split")
        },
        artifacts_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    )
    (output / "labram.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def plot_comparison(frame: pd.DataFrame, output: Path) -> None:
    """Plot retrieval metrics without equating waveform correlations and embedding cosines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = ["waveform", "corrca", "misaligned_fit"] + sorted(
        m for m in frame.model.unique() if m.startswith("labram_")
    )
    for ax, metric, chance in zip(axes, ["top1", "mrr"], [0.25, sum(1 / np.arange(1, 5)) / 4], strict=True):
        for split, rows in frame.groupby("split"):
            values = rows.groupby("model")[metric].mean().reindex(order)
            ax.plot(values.index, values.values, marker="o", label=split)
        ax.axhline(chance, color="gray", linestyle="--", label="chance")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend()
    fig.suptitle("Fixed video galleries: frozen representations (not training curves)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"comparison.{extension}", dpi=160)
    plt.close(fig)
