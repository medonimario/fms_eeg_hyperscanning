"""Reproducible fold execution, prediction exports, and cross-validation summaries."""

from __future__ import annotations

import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from .data import WindowDataset, fold_datasets, sha256
from .model import load_model
from .tracking import WandbProgress, log_result, tracked_run
from .training import make_classifier


def write_json(path: Path, value: dict) -> None:
    """Write readable JSON, refusing non-finite metrics."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metrics(target: np.ndarray, probability: np.ndarray) -> dict:
    """Compute window-level metrics with a fixed 0.5 decision threshold."""
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid prediction probabilities.")
    predicted = (probability > 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(target, probability)
    return dict(
        accuracy=float(accuracy_score(target, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(target, predicted)),
        roc_auc=float(roc_auc_score(target, probability)),
        pr_auc=float(auc(recall, precision)),
        average_precision=float(average_precision_score(target, probability)),
        f1=float(f1_score(target, predicted, zero_division=0)),
        confusion_matrix=confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
    )


def run_fold(cache: Path, output: Path, checkpoint: Path, fold: int, options: dict) -> dict:
    """Fit one fold, restore its validation-selected model, and evaluate test once.

    Args:
        cache: Complete prepared dataset cache.
        output: New directory for this fold; existing results are never overwritten.
        checkpoint: Original local pretraining checkpoint.
        fold: Held-out test fold in 0 through 9.
        options: Validated training options from the CLI.

    Returns:
        Test metrics and the selected epoch, also saved in result.json.
    """
    cache_config, datasets = fold_datasets(cache, fold)
    options = dict(options)
    if options["lr"] is None:
        options["lr"] = 1e-4 if cache_config["dataset"] == "mg" else 5e-4
    if options["smoke_windows_per_class"]:
        for role, dataset in datasets.items():
            rows = dataset.metadata.groupby("target", sort=True).head(options["smoke_windows_per_class"])
            datasets[role] = WindowDataset(cache, rows)
    steps = len(datasets["train"]) // (options["batch_size"] * options["update_freq"])
    if steps < 1:
        raise ValueError("Training partition is too small for one full batch-size × update-freq update.")
    if options["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the environment or use --device cpu.")
    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options["seed"])
    torch.backends.cudnn.benchmark = False
    model = load_model(checkpoint, options["drop_path"])
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        cache=cache_config,
        checkpoint_sha256=sha256(checkpoint),
        fold=fold,
        options=options,
        steps_per_epoch=steps,
        versions={name: version(name) for name in ("torch", "braindecode", "mne", "skorch", "numpy", "scikit-learn")},
    )
    write_json(output / "config.json", config)
    for role, dataset in datasets.items():
        dataset.metadata.to_csv(output / f"{role}_windows.csv", index=False)
        print(
            f"{role}: {len(dataset)} windows, {dataset.metadata.group.nunique()} groups; "
            f"class counts {dataset.metadata.target.value_counts().to_dict()}",
            flush=True,
        )
    with tracked_run(config, output) as run:
        net = make_classifier(model, datasets["valid"], output, options)
        if run is not None:
            net.set_params(
                callbacks=net.callbacks + [("wandb", WandbProgress(run, options.get("wandb_log_every", 20)))]
            )
        net.fit(datasets["train"], y=None)
        net.history.to_file(str(output / "history.json"))
        best = max(net.history, key=lambda row: row["valid_balanced_accuracy"])
        result = dict(fold=fold, best_epoch=best["epoch"], smoke=bool(options["smoke_windows_per_class"]))
        for role in ("valid", "test"):
            probability = net.predict_proba(datasets[role])[:, 1]
            rows = datasets[role].metadata.copy()
            result[role] = metrics(rows.target.to_numpy(), probability)
            rows["probability_class_1"] = probability
            rows["prediction"] = (probability > 0.5).astype(int)
            rows.to_csv(output / f"{role}_predictions.csv", index=False)
        log_result(run, result)
    result["config_sha256"] = sha256(output / "config.json")
    write_json(output / "result.json", result)
    print(f"Completed fold {fold}: {result['test']}", flush=True)
    return result


def summarize(directory: Path) -> dict:
    """Require ten comparable completed folds and report mean/sample SD of test metrics."""
    results, signatures = [], []
    for fold in range(10):
        folder = directory / f"fold_{fold}"
        result = json.loads((folder / "result.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        if result["fold"] != fold or config["fold"] != fold or result["smoke"]:
            raise ValueError("Summary requires all ten distinct, full-data folds.")
        if sha256(folder / "config.json") != result["config_sha256"]:
            raise ValueError("Run configuration changed after completion.")
        signatures.append({key: value for key, value in config.items() if key not in {"fold", "steps_per_epoch"}})
        results.append(result)
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError("Fold configurations differ; do not combine these runs.")
    scores = pd.DataFrame(
        [{key: value for key, value in row["test"].items() if key != "confusion_matrix"} for row in results]
    )
    scores.index.name = "fold"
    scores.to_csv(directory / "fold_metrics.csv")
    summary = {key: dict(mean=float(scores[key].mean()), sample_std=float(scores[key].std(ddof=1))) for key in scores}
    write_json(directory / "summary.json", summary)
    return summary
