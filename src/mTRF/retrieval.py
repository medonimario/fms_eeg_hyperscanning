"""Identify held-out movement trials from EEG using the original full mTRF models."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import eeg
from .common import CONDITIONS, PAIRS, dataset_root, save_pickle

METRICS = ("top1", "mrr", "chance_top1", "chance_mrr")


def sha256(path: Path) -> str:
    """Hash a source file without reading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def retrieval_scores(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Score EEG queries against movement-generated EEG with channel-averaged residual R².

    Args:
        observed: EEG queries with shape (trials, samples, channels).
        predicted: Predicted EEG for the candidate movement trials with the same shape.

    Returns:
        Matrix indexed by query trial and candidate trial; larger values rank higher.
    """
    variance = ((observed - observed.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    if (variance <= 0).any() or not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Retrieval requires finite signals and nonconstant query EEG channels.")
    return np.array(
        [
            np.mean(1 - ((predicted - query) ** 2).sum(axis=1) / total, axis=1)
            for query, total in zip(observed, variance, strict=True)
        ]
    )


def query_metrics(scores: np.ndarray, target: int) -> dict[str, float]:
    """Return expected top-1 and reciprocal rank under uniform breaking of exact ties."""
    if len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("A retrieval gallery needs at least two finite candidate scores.")
    better = int(np.sum(scores > scores[target]))
    tied = int(np.sum(scores == scores[target]))
    return {
        "top1": float(1 / tied if better == 0 else 0),
        "mrr": float(np.mean(1 / np.arange(better + 1, better + tied + 1))),
        "chance_top1": 1 / len(scores),
        "chance_mrr": float(np.mean(1 / np.arange(1, len(scores) + 1))),
    }


def evaluate_fold(features: np.ndarray, observed: np.ndarray, keep: list[int], reference: tuple, fold: int) -> dict:
    """Refit using the saved penalty, verify original scores and rank the outer test trials."""
    splits = np.array_split(np.random.default_rng(123).permutation(len(keep)), 5)
    test = splits[fold]
    train = np.concatenate(splits[:fold] + splits[fold + 1 :])
    regularization = float(reference[1][fold])
    model = eeg.TRF(direction=1, preload=False)
    model.train(list(features[train]), list(observed[train]), eeg.FS, *eeg.LAGS, regularization)
    prediction, correlation, rsq = model.predict(list(features[test]), list(observed[test]))
    expected = np.array([reference[0][fold], reference[2][fold]], dtype=float)
    actual = np.array([correlation, rsq], dtype=float)
    if not np.isfinite(expected).all() or not np.allclose(actual, expected, rtol=1e-5, atol=1e-8):
        raise ValueError(
            f"Fold {fold + 1} does not reproduce original correlation/R²: {actual} versus {expected}. "
            "Check EEG, phase, artifact dictionary, fork version and trial ordering."
        )
    scores = retrieval_scores(observed[test], np.asarray(prediction))
    if not np.isclose(np.diag(scores).mean(), rsq, rtol=1e-5, atol=1e-8):
        raise ValueError("Retrieval diagonal disagrees with original fold R².")
    return {
        "fold": fold + 1,
        "train_trials": np.asarray(keep)[train].tolist(),
        "test_trials": np.asarray(keep)[test].tolist(),
        "regularization": regularization,
        "reference_scores": expected.tolist(),
        "refit_scores": actual.tolist(),
        "scores": scores,
    }


def evaluate_pair(root: Path, pair: int, phase: tuple, bad: dict, reference: dict, jobs: int) -> list[dict]:
    """Evaluate the full model for each participant, condition and original outer fold."""
    results = []
    for condition, member, keep, features, observed in eeg.participant_trials(root, pair, phase, bad):
        saved = reference[f"{condition}_Full"][2 * PAIRS.index(pair) + member]
        if any(len(values) != 5 for values in saved):
            raise ValueError("Original scores must contain five correlations, penalties and R² values.")
        folds = Parallel(n_jobs=jobs)(
            delayed(evaluate_fold)(features, observed, keep, saved, fold) for fold in range(5)
        )
        for result in folds:
            result.update(pair=pair, member="ab"[member], condition=condition)
            results.append(result)
    return results


def collect(directory: Path) -> None:
    """Collect all dyads, summarize retrieval versus analytic chance and save a figure."""
    rows, matrices = [], []
    protocol = None
    for pair in PAIRS:
        with (directory / f"pair{pair}.pickle").open("rb") as stream:
            shard = pickle.load(stream)
        if protocol is None:
            protocol = shard["protocol"]
        if shard["protocol"] != protocol or shard["pair"] != pair:
            raise ValueError(f"Incompatible retrieval shard for pair {pair}. Use a separate directory per run.")
        folds = shard["folds"]
        expected_keys = {
            (condition, member, fold) for condition in CONDITIONS for member in "ab" for fold in range(1, 6)
        }
        if (
            len(folds) != len(expected_keys)
            or {(f["condition"], f["member"], f["fold"]) for f in folds} != expected_keys
        ):
            raise ValueError(f"Missing or duplicated folds for pair {pair}.")
        for result in folds:
            scores = result["scores"]
            labels = result["test_trials"]
            if scores.shape != (len(labels), len(labels)):
                raise ValueError(f"Invalid retrieval score matrix for pair {pair}.")
            for query, trial in enumerate(labels):
                identifiers = dict(
                    pair=pair,
                    member=result["member"],
                    condition=result["condition"],
                    fold=result["fold"],
                    query_trial=trial,
                    gallery_size=len(labels),
                )
                rows.append({**identifiers, **query_metrics(scores[query], query)})
                for candidate, candidate_trial in enumerate(labels):
                    matrices.append(
                        {
                            **identifiers,
                            "candidate_trial": candidate_trial,
                            "is_match": query == candidate,
                            "r2": scores[query, candidate],
                        }
                    )
    queries = pd.DataFrame(rows)
    queries.to_csv(directory / "queries.csv", index=False)
    pd.DataFrame(matrices).to_csv(directory / "candidate_scores.csv", index=False)
    participants = queries[queries.pair != 2020].groupby(["pair", "member", "condition"])[list(METRICS)].mean()
    participants.to_csv(directory / "participants.csv")
    dyads = participants.groupby(["pair", "condition"]).mean()
    dyads["top1_minus_chance"] = dyads.top1 - dyads.chance_top1
    dyads["mrr_minus_chance"] = dyads.mrr - dyads.chance_mrr
    dyads.to_csv(directory / "dyads.csv")
    summary = dyads.groupby("condition").mean().reindex(CONDITIONS)
    summary["dyads"] = dyads.groupby("condition").size()
    summary.to_csv(directory / "summary.csv")
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    plot_summary(directory, summary)
    print(summary.to_string())


def plot_summary(directory: Path, summary: pd.DataFrame) -> None:
    """Render condition-level mean retrieval and matched-gallery chance."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    positions = np.arange(len(summary))
    for ax, metric, label in zip(axes, ("top1", "mrr"), ("Top-1 accuracy", "Mean reciprocal rank"), strict=True):
        ax.bar(positions - 0.18, summary[metric], width=0.36, label="mTRF")
        ax.bar(positions + 0.18, summary[f"chance_{metric}"], width=0.36, label="Chance")
        ax.set_xticks(positions, [eeg.LABELS[c] for c in summary.index])
        ax.set(ylabel=label, ylim=(0, 1))
        ax.legend()
    fig.savefig(directory / "retrieval.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run independent dyad jobs or collect a completed retrieval array."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--data-dir", type=Path)
    run.add_argument("--phase", type=Path, required=True)
    run.add_argument("--run", type=Path, required=True, help="Completed mTRF run with combined fold scores.")
    run.add_argument("--bad-segments", type=Path)
    run.add_argument("--output", type=Path, required=True, help="Separate retrieval run directory.")
    run.add_argument("--jobs", type=int, default=1)
    group = run.add_mutually_exclusive_group()
    group.add_argument("--pair", type=int, choices=PAIRS)
    group.add_argument("--pair-index", type=int, choices=range(1, len(PAIRS) + 1))
    merge = commands.add_parser("collect")
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output)
        return
    root = dataset_root(args.data_dir)
    if args.output.resolve() == args.run.resolve():
        raise ValueError("Use a separate retrieval output directory to protect the original mTRF results.")
    bad_path = args.bad_segments or root / "dict_with_short_bad_epochs.pickle"
    reference_path = args.run / eeg.CV_FILE
    protocol = {
        "version": 1,
        "model": "Full",
        "score": "mean_channel_residual_r2",
        "seed": 123,
        "folds": 5,
        "gallery": "all held-out whole trials of the same participant, condition and outer fold",
        "trial_indices": "zero-based aligned condition trials after pair2450 correction",
        "ties": "uniform random order within exact ties",
        "summary_excludes": [2020],
        "chance": "uniform candidate ranking; per-query chance averaged like model metrics",
        "phase_sha256": sha256(args.phase),
        "bad_segments_sha256": sha256(bad_path),
        "reference_sha256": sha256(reference_path),
        "code_sha256": {
            name: sha256(Path(__file__).parent / name)
            for name in (
                "eeg.py",
                "retrieval.py",
                "mtrf_modified/model.py",
                "mtrf_modified/matrices.py",
                "mtrf_modified/stats.py",
            )
        },
    }
    with args.phase.open("rb") as stream:
        phases = pickle.load(stream)
    with bad_path.open("rb") as stream:
        bad = pickle.load(stream)
    with reference_path.open("rb") as stream:
        reference = pickle.load(stream)
    if len(phases) != len(PAIRS):
        raise ValueError("Expected the original ordered 52-pair phase list.")
    pair = PAIRS[args.pair_index - 1] if args.pair_index else args.pair
    args.output.mkdir(parents=True, exist_ok=True)
    for selected in (pair,) if pair else PAIRS:
        print(f"Retrieving movement for pair {selected}", flush=True)
        folds = evaluate_pair(root, selected, phases[PAIRS.index(selected)], bad, reference, args.jobs)
        sources = {
            member: {
                "path": str(eeg.eeg_path(root, selected, member)),
                "sha256": sha256(eeg.eeg_path(root, selected, member)),
            }
            for member in "ab"
        }
        save_pickle(
            args.output / f"pair{selected}.pickle",
            {"pair": selected, "protocol": protocol, "sources": sources, "folds": folds},
        )
    if pair is None:
        collect(args.output)


if __name__ == "__main__":
    main()
