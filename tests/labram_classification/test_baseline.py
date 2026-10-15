"""Validate spectral features and leakage-free classical baseline selection."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from labram_classification.baseline import bandpower_features, feature_names, run_baseline, select_classifier
from labram_classification.data import CHANNELS, fold_datasets, sha256


def test_band_power_and_units() -> None:
    """A 2-microvolt 10-Hz sinusoid has 2 microvolts squared of alpha power."""
    time = np.arange(800) / 200
    signal = np.broadcast_to(2 * np.sin(2 * np.pi * 10 * time), (2, 64, 800))
    features = bandpower_features(signal).reshape(2, 64, 5)
    np.testing.assert_allclose(10 ** features[..., 2], 2, rtol=1e-6)
    stronger = bandpower_features(signal * 10).reshape(2, 64, 5)
    np.testing.assert_allclose(stronger[..., 2] - features[..., 2], 2, rtol=1e-6)
    assert np.isfinite(bandpower_features(np.zeros((1, 64, 800)))).all()
    assert len(feature_names("bandpower")) == 320


def test_time_resolved_features_and_welch_average() -> None:
    """Preserve early alpha/late beta changes while whole-window power averages them."""
    time = np.arange(800) / 200
    signal = np.where(time < 2, np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 20 * time))
    signal = np.broadcast_to(signal, (1, 64, 800))
    temporal = bandpower_features(signal, "time-bandpower").reshape(1, 3, 64, 5)
    whole = bandpower_features(signal).reshape(1, 64, 5)
    assert temporal[0, 0, 0, 2] > temporal[0, 0, 0, 3]
    assert temporal[0, 2, 0, 3] > temporal[0, 2, 0, 2]
    np.testing.assert_allclose(10**whole, (10**temporal).mean(axis=1), rtol=2e-6, atol=1e-12)
    assert len(feature_names("time-bandpower")) == 960


def test_scaler_fits_training_only_and_ties_choose_small_c() -> None:
    """Extreme validation values cannot shift the learned feature normalization."""
    training = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    validation = np.array([[100.0, 100.0], [200.0, 200.0]])
    model, candidates = select_classifier(training, labels, validation, np.array([0, 1]), [1.0, 0.01], 0, 1000)
    np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, training.mean(axis=0))
    assert model.named_steps["logisticregression"].C == 0.01
    assert candidates.C.tolist() == [0.01, 1.0]
    assert not any("test" in column for column in candidates)


def make_cache(directory: Path) -> None:
    """Create 40 windows across ten independent groups for an end-to-end CV test."""
    directory.mkdir()
    random = np.random.RandomState(1)
    time = np.arange(800) / 200
    signals, rows = [], []
    for group in range(10):
        for window in range(4):
            target = window % 2
            signal = random.normal(scale=0.2, size=(64, 800))
            signal += np.sin(2 * np.pi * (10 if target else 20) * time)
            signals.append(signal)
            rows.append(
                dict(
                    group=group,
                    participant=str(group),
                    fold=group,
                    target=target,
                    array="signal.npy",
                    array_index=len(rows),
                    source=f"group_{group}",
                )
            )
    np.save(directory / "signal.npy", np.asarray(signals, dtype=np.float32))
    pd.DataFrame(rows).to_csv(directory / "windows.csv", index=False)
    config = dict(
        format_version=1,
        channels=CHANNELS,
        dataset="mg",
        contrast="solo-coordination",
        sfreq=200,
        n_times=800,
        stored_units="microvolts",
        metadata_sha256=sha256(directory / "windows.csv"),
    )
    (directory / "cache.json").write_text(json.dumps(config))


@pytest.mark.parametrize("feature_mode", ["bandpower", "time-bandpower"])
def test_complete_baseline_cv(tmp_path: Path, feature_mode: str) -> None:
    """Train ten folds, match LaBraM partitions, and verify saved estimator predictions."""
    cache, output = tmp_path / "cache", tmp_path / "baseline"
    make_cache(cache)
    options = dict(
        feature_mode=feature_mode,
        feature_batch_size=7,
        num_threads=1,
        max_iter=2000,
        c_values=[0.01, 0.1],
        seed=0,
        smoke_windows_per_class=0,
        wandb_mode="disabled",
    )
    run_baseline(cache, output, options)
    assert (output / "summary.json").exists()
    assert len(pd.read_csv(output / "fold_metrics.csv")) == 10
    original = pd.read_csv(output / "feature_windows.csv")
    features = np.load(output / "features.npy")
    for fold in range(10):
        folder = output / f"fold_{fold}"
        _, reference = fold_datasets(cache, fold)
        for role in ("train", "valid", "test"):
            actual = pd.read_csv(folder / f"{role}_windows.csv", dtype={"participant": str})
            pd.testing.assert_frame_equal(actual, reference[role].metadata)
        test = pd.read_csv(folder / "test_predictions.csv")
        model = joblib.load(folder / "model.joblib")
        indices = original.index[original.fold == fold]
        np.testing.assert_allclose(model.predict_proba(features[indices])[:, 1], test.probability_class_1)
        result = json.loads((folder / "result.json").read_text())
        assert result["test"]["balanced_accuracy"] == 1
        assert "best_C" in result and "best_epoch" not in result
    with pytest.raises(FileExistsError):
        run_baseline(cache, output, options)
