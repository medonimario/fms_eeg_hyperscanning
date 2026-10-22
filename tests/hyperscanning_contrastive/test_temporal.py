"""Temporal bridge checks for split isolation, real second windows, and matched runs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_ssl import TinyEncoder, make_cache

from hyperscanning_contrastive import temporal
from hyperscanning_contrastive.data import sha256
from hyperscanning_contrastive.ssl import EEGCache, SSLConfig


def temporal_cache(path: Path) -> Path:
    """Add distinct second windows to the existing three-fold fixture."""
    make_cache(path)
    metadata = json.loads((path / "cache.json").read_text())
    rows = pd.read_csv(path / "windows.csv")
    targets = rows.assign(crop_start_s=4, array_index=rows.array_index + 12)
    for name in metadata["array_sha256"]:
        values = np.load(path / name)
        np.save(path / name, np.concatenate([values, values + 1]))
        metadata["array_sha256"][name] = sha256(path / name)
    pd.concat([rows, targets]).to_csv(path / "windows.csv", index=False)
    metadata["manifest_sha256"]["windows.csv"] = sha256(path / "windows.csv")
    (path / "cache.json").write_text(json.dumps(metadata))
    return path


def test_windows_and_split(tmp_path: Path) -> None:
    """Both persons share whole-trial assignments and reserved dyads never enter extraction."""
    cache = EEGCache(temporal_cache(tmp_path / "cache"), SSLConfig(pairs=2))
    rows, census = temporal.temporal_trials(cache, 2, 2042)
    assert set(rows.fold) == {0}
    assert set(census.eligible_trials) == {4}
    assert not set(rows[rows.split == "train"].trial_id) & set(rows[rows.split == "validation"].trial_id)
    bank = temporal.temporal_features(TinyEncoder("random", None, False), cache, rows)
    for split, entries in bank.items():
        for key, entry in entries.items():
            assert key[0] == 0
            assert entry["split"] == split
            assert not torch.equal(entry["a"], entry["b"])
            other = entries[(*key[:2], 1 - key[2])]
            assert entry["rows"].trial_id.tolist() == other["rows"].trial_id.tolist()


def test_missing_windows_fails(tmp_path: Path) -> None:
    """Missing target windows fail before any modeling rather than silently dropping strata."""
    cache = EEGCache(make_cache(tmp_path / "cache"), SSLConfig(pairs=2))
    with pytest.raises(ValueError, match="clean temporal pairs"):
        temporal.temporal_trials(cache, 2, 2042)


def test_matched_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run both arms and temperatures with full coverage and validate checkpoint selection."""
    monkeypatch.setattr(temporal, "ContrastiveLabram", TinyEncoder)
    path = temporal_cache(tmp_path / "cache")
    output = tmp_path / "run"
    report = temporal.run_temporal(
        path, output, tmp_path / "absent", initialization="random", pairs=2, epochs=2, temperatures=(0.05, 0.2)
    )
    assert report["checkpoint_sha256"] is None
    assert report["pair_presentations_per_epoch"] == 12
    assert report["updates_per_epoch"] == 2
    assert report["negatives_per_training_query_range"] == [1, 1]
    splits = pd.read_csv(output / "trial_splits.csv")
    mapping = pd.read_csv(output / "shuffled_training_pairs.csv")
    assert set(mapping.target_trial) <= set(splits[splits.split == "train"].trial_id)
    for label, arms in report["experiments"].items():
        histories = []
        for arm, result in arms.items():
            history = pd.read_csv(output / label / arm / "history.csv")
            assert result["best_epoch"] == history.mrr.idxmax()
            assert history.iloc[0].mrr == pytest.approx(report["baselines"]["initial_projection"]["macro"]["mrr"])
            histories.append(history)
        assert histories[0].iloc[0].mrr == histories[1].iloc[0].mrr
    assert (output / "temporal_curves.png").stat().st_size > 1000


@pytest.mark.parametrize(
    "options",
    [{"learning_rates": (0,)}, {"weight_decays": (-1,)}, {"learning_rates": (float("nan"),)}, {"weight_decays": ()}],
)
def test_invalid_optimization(tmp_path: Path, options: dict) -> None:
    """Reject invalid sweep values before loading data or creating output."""
    with pytest.raises(ValueError, match="Invalid"):
        temporal.run_temporal(tmp_path / "missing", tmp_path / "output", tmp_path / "missing", **options)


def test_optimization_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each grid point gets its optimizer settings and matched initial metrics."""
    monkeypatch.setattr(temporal, "ContrastiveLabram", TinyEncoder)
    calls = []
    original = torch.optim.AdamW

    def optimizer(parameters, **kwargs):
        """Record settings actually passed to the optimizer."""
        calls.append(kwargs)
        return original(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    monkeypatch.setattr(temporal, "plot_temporal", lambda _: None)
    report = temporal.run_temporal(
        temporal_cache(tmp_path / "cache"),
        tmp_path / "out",
        tmp_path / "absent",
        initialization="random",
        pairs=2,
        epochs=1,
        temperatures=(0.05,),
        learning_rates=(0.001, 0.0003),
        weight_decays=(0.01, 0.1),
    )
    assert len(report["experiments"]) == 4
    assert len(calls) == 8
    assert {(c["lr"], c["weight_decay"]) for c in calls} == {(lr, wd) for lr in (0.001, 0.0003) for wd in (0.01, 0.1)}
    assert report["selected_settings"] == report["experiment_settings"][report["selected_experiment"]]
    assert len(pd.read_csv(tmp_path / "out" / "comparison.csv")) == 8
