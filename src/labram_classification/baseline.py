"""Log-band-power logistic-regression baselines on the existing grouped folds."""

from __future__ import annotations

import json
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.signal import get_window, welch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import CHANNELS, fold_datasets, sha256
from .runner import metrics, summarize, write_json
from .tracking import log_result, tracked_run

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "low_gamma": (30, 45)}
MODES = ("bandpower", "time-bandpower")
POWER_FLOOR = 1e-12


def feature_names(mode: str) -> list[str]:
    """Name flattened features in time-bin, channel, then frequency-band order."""
    if mode not in MODES:
        raise ValueError(f"Feature mode must be one of {MODES}.")
    bins = ["whole"] if mode == "bandpower" else ["0-2s", "1-3s", "2-4s"]
    return [f"{time_bin}/{channel}/{band}" for time_bin in bins for channel in CHANNELS for band in BANDS]


def bandpower_features(signal: np.ndarray, mode: str = "bandpower") -> np.ndarray:
    """Extract fixed, label-free log10 absolute power features from microvolt windows.

    Args:
        signal: Finite EEG shaped (windows, 64 channels, 800 samples), at 200 Hz.
        mode: bandpower averages periodograms over the window; time-bandpower
            retains the three overlapping two-second periodograms separately.

    Returns:
        Float32 feature matrix with 320 or 960 columns. Power before log10 is in
        microvolts squared; a fixed 1e-12 floor prevents log(0).
    """
    feature_names(mode)
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 3 or signal.shape[1:] != (64, 800) or not np.isfinite(signal).all():
        raise ValueError("Expected finite microvolt EEG shaped (windows, 64, 800).")
    segments = [signal] if mode == "bandpower" else [signal[..., start : start + 400] for start in (0, 200, 400)]
    output = []
    for segment in segments:
        frequencies, psd = welch(
            segment,
            fs=200,
            window=get_window("hann", 400, fftbins=True),
            nperseg=400,
            noverlap=200,
            nfft=400,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            average="mean",
            axis=-1,
        )
        powers = []
        for low, high in BANDS.values():
            indices = (frequencies >= low) & (frequencies <= high)
            powers.append(np.trapezoid(psd[..., indices], frequencies[indices], axis=-1))
        values = np.log10(np.maximum(np.stack(powers, axis=-1), POWER_FLOOR))
        output.append(values.reshape(len(signal), -1))
    return np.concatenate(output, axis=1).astype(np.float32)


def extract_cache_features(
    cache: Path, destination: Path, mode: str, batch_size: int = 64
) -> tuple[dict, pd.DataFrame]:
    """Extract each recording in bounded chunks, preserving original window row order."""
    if batch_size < 1:
        raise ValueError("Feature batch size must be positive.")
    config, _ = fold_datasets(cache, 0)
    if config.get("sfreq") != 200 or config.get("n_times") != 800 or config.get("stored_units") != "microvolts":
        raise ValueError("Baseline requires the prepared 200-Hz, 800-sample microvolt cache.")
    metadata = pd.read_csv(cache / "windows.csv", dtype={"participant": str})
    if metadata.duplicated(["array", "array_index"]).any():
        raise ValueError("Cache contains duplicate window identities.")
    names = feature_names(mode)
    features = np.lib.format.open_memmap(
        destination / "features.npy", mode="w+", dtype=np.float32, shape=(len(metadata), len(names))
    )
    groups = list(metadata.groupby("array", sort=False))
    for recording_index, (filename, rows) in enumerate(groups, start=1):
        path = (cache / filename).resolve()
        if not path.is_relative_to(cache.resolve()):
            raise ValueError("Array path escapes cache directory.")
        signal = np.load(path, mmap_mode="r", allow_pickle=False)
        print(f"Features [{recording_index}/{len(groups)}]: {filename}", flush=True)
        indices = rows.array_index.to_numpy()
        if not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any() or (indices >= len(signal)).any():
            raise ValueError(f"Invalid window indices for {filename}.")
        for start in range(0, len(rows), batch_size):
            subset = rows.iloc[start : start + batch_size]
            features[subset.index] = bandpower_features(signal[subset.array_index.to_numpy()], mode)
    features.flush()
    metadata.to_csv(destination / "feature_windows.csv", index=False)
    pd.DataFrame({"feature": names}).to_csv(destination / "feature_names.csv", index=False)
    specification = dict(
        version=1,
        mode=mode,
        bands=BANDS,
        sfreq=200,
        n_times=800,
        segment_samples=400,
        segment_stride=200,
        window="periodic Hann",
        detrend="constant",
        units="microvolts squared",
        transform="log10(max(power, 1e-12))",
        power_floor=POWER_FLOOR,
        n_features=len(names),
        cache_metadata_sha256=config["metadata_sha256"],
        feature_matrix_sha256=sha256(destination / "features.npy"),
    )
    write_json(destination / "features.json", specification)
    return specification, metadata


def select_classifier(
    training: np.ndarray,
    labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    c_values: list[float],
    seed: int,
    max_iter: int,
    run: Any = None,
) -> tuple[Any, pd.DataFrame]:
    """Fit only on training data and select C by validation balanced accuracy.

    The ascending C grid breaks ties in favor of stronger regularization. No
    test data are accepted by this function. A nonconverged fit fails explicitly.
    """
    if not c_values or any(not np.isfinite(value) or value <= 0 for value in c_values):
        raise ValueError("All C values must be finite and positive.")
    best, best_score, rows = None, -np.inf, []
    for candidate, value in enumerate(sorted(set(c_values)), start=1):
        estimator = make_pipeline(
            StandardScaler(), LogisticRegression(C=value, solver="lbfgs", max_iter=max_iter, random_state=seed)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(training, labels)
        train_probability = estimator.predict_proba(training)[:, 1]
        valid_probability = estimator.predict_proba(validation)[:, 1]
        score = balanced_accuracy_score(validation_labels, valid_probability > 0.5)
        row = dict(
            candidate=candidate,
            C=value,
            train_loss=log_loss(labels, train_probability, labels=[0, 1]),
            valid_loss=log_loss(validation_labels, valid_probability, labels=[0, 1]),
            train_balanced_accuracy=balanced_accuracy_score(labels, train_probability > 0.5),
            valid_balanced_accuracy=score,
        )
        rows.append(row)
        print(f"C={value:g}: validation balanced accuracy={score:.4f}", flush=True)
        if run is not None:
            run.log({f"selection/{key}": item for key, item in row.items()})
        if score > best_score:
            best, best_score = estimator, score
    return best, pd.DataFrame(rows)


def run_baseline_fold(
    cache: Path,
    output: Path,
    fold: int,
    features: np.ndarray,
    metadata: pd.DataFrame,
    specification: dict,
    options: dict,
) -> dict:
    """Reuse LaBraM partitions, fit a linear baseline, and export comparable test metrics."""
    cache_config, datasets = fold_datasets(cache, fold)
    if cache_config["metadata_sha256"] != specification["cache_metadata_sha256"]:
        raise ValueError("EEG cache metadata changed after feature extraction.")
    identities = pd.MultiIndex.from_frame(metadata[["array", "array_index"]])
    partitions = {}
    for role, dataset in datasets.items():
        indices = identities.get_indexer(pd.MultiIndex.from_frame(dataset.metadata[["array", "array_index"]]))
        if (indices < 0).any():
            raise ValueError("Feature cache and fold window identities disagree.")
        partitions[role] = (np.asarray(features[indices], dtype=np.float64), dataset.metadata.target.to_numpy())
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        model="logistic_regression",
        classifier=dict(
            solver="lbfgs",
            penalty="l2",
            class_weight=None,
            threshold=0.5,
            preprocessing="StandardScaler fitted on training only",
        ),
        fold=fold,
        cache=cache_config,
        features=specification,
        options=options,
        versions={name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas", "joblib")},
    )
    write_json(output / "config.json", config)
    for role, dataset in datasets.items():
        dataset.metadata.to_csv(output / f"{role}_windows.csv", index=False)
    with tracked_run(config, output) as run:
        model, candidates = select_classifier(
            *partitions["train"], *partitions["valid"], options["c_values"], options["seed"], options["max_iter"], run
        )
        candidates.to_csv(output / "selection.csv", index=False)
        joblib.dump(model, output / "model.joblib")
        pd.DataFrame(
            {
                "feature": feature_names(options["feature_mode"]),
                "standardized_coefficient": model.named_steps["logisticregression"].coef_[0],
            }
        ).to_csv(output / "coefficients.csv", index=False)
        result = dict(fold=fold, smoke=False, best_C=float(model.named_steps["logisticregression"].C))
        for role in ("train", "valid", "test"):
            signal, target = partitions[role]
            probability = model.predict_proba(signal)[:, 1]
            result[role] = metrics(target, probability)
            result[role]["loss"] = float(log_loss(target, probability, labels=[0, 1]))
            rows = datasets[role].metadata.copy()
            rows["probability_class_1"] = probability
            rows["prediction"] = (probability > 0.5).astype(int)
            rows.to_csv(output / f"{role}_predictions.csv", index=False)
        log_result(run, result)
    result["config_sha256"] = sha256(output / "config.json")
    write_json(output / "result.json", result)
    print(f"Completed baseline fold {fold}: {result['test']}", flush=True)
    return result


def run_baseline(cache: Path, output: Path, options: dict, fold: int | None = None) -> None:
    """Extract deterministic features once and run one fold or the full ten-fold experiment."""
    if output.exists():
        raise FileExistsError("Use a new baseline output directory.")
    if (
        options["feature_mode"] not in MODES
        or min(options["max_iter"], options["num_threads"], options["feature_batch_size"]) < 1
    ):
        raise ValueError("Invalid baseline mode or resource settings.")
    if not options["c_values"] or any(not np.isfinite(c) or c <= 0 for c in options["c_values"]):
        raise ValueError("C values must be finite and positive.")
    if fold is not None and not 0 <= fold < 10:
        raise ValueError("Fold must be in 0..9.")
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=options["num_threads"]):
        specification, metadata = extract_cache_features(
            cache, output, options["feature_mode"], options["feature_batch_size"]
        )
        features = np.load(output / "features.npy", mmap_mode="r", allow_pickle=False)
        for selected in range(10) if fold is None else [fold]:
            run_baseline_fold(cache, output / f"fold_{selected}", selected, features, metadata, specification, options)
    if fold is None:
        print(json.dumps(summarize(output), indent=2))
