"""Alignment, reserved-data and bounded extraction regression checks."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_splits import cache  # noqa: F401
from torch import nn

from cl_behaviour.features import encode_windows, extract_features, feature_plan, phase_features, verified_array
from cl_behaviour.splits import split_cache
from hyperscanning_contrastive.data import sha256
from mTRF.common import PAIRS


class FakeTokens(nn.Module):
    """Expose sample identity in tokens to test ordering without a foundation model."""

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Expand one identifying sample into the token grid."""
        return signal[:, :1, :1].reshape(-1, 1, 1, 1).expand(-1, 64, 4, 200)


def make_run(cache, tmp_path):
    """Write a genuine split completion with a synthetic historical cohort."""
    cohort = tmp_path / "cohort.json"
    folds = {group: i % 10 for i, group in enumerate(PAIRS)}
    cohort.write_text(json.dumps(dict(dataset="circling-local-project1", group_folds=folds)))
    run = tmp_path / "split"
    split_cache(cache, cohort, run)
    return run


def test_phase_wrap_alignment_and_antialias():
    """Preserve phase across wraps and suppress signals above the output Nyquist limit."""
    time = np.arange(4320) / 240
    phase = np.angle(np.exp(1j * (2 * np.pi * 2 * time + 0.7)))
    result = phase_features(phase)
    expected = 2 * np.pi * 2 * np.arange(900) / 50 + 0.7
    np.testing.assert_allclose(result[:, 20:-20], np.array([np.sin(expected), np.cos(expected)])[:, 20:-20], atol=0.003)
    fast = phase_features(2 * np.pi * 60 * time)
    assert np.abs(fast[:, 40:-40]).max() < 0.01
    assert result.shape == (2, 900) and result.dtype == np.float32
    for start in range(0, 15, 2):
        assert result[:, start * 50 : (start + 4) * 50].shape == (2, 200)
    with pytest.raises(ValueError, match="Invalid"):
        phase_features(np.full(4320, np.nan))


def test_plan_excludes_test_reserved_and_open(cache, tmp_path):
    """Restrict features before any signal access and keep all eight ordered windows."""
    run = make_run(cache, tmp_path)
    _, _, trials, windows = feature_plan(cache, run)
    assert set(trials.split) == {"train", "validation"}
    assert not trials.cohort_fold.isin([1, 2]).any()
    assert set(trials.condition) == {"closed1", "closed2"}
    assert set(windows.trial_id) == set(trials.trial_id)
    assert windows.groupby("trial_id").size().eq(8).all()
    _, _, tiny, tiny_windows = feature_plan(cache, run, smoke=True)
    assert len(tiny) == 2 and len(tiny_windows) == 16
    assert tiny.group.nunique() == 1 and set(tiny.split) == {"train"}
    with (run / "windows.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Split hash mismatch"):
        feature_plan(cache, run)


def test_streaming_preserves_members_and_ignores_unselected_nan(tmp_path):
    """Never materialize excluded rows and preserve paired input indexing across batches."""
    signals = np.full((4, 2, 64, 800), np.nan, dtype=np.float32)
    signals[1, 0], signals[1, 1] = 10, 11
    signals[3, 0], signals[3, 1] = 30, 31
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    encode_windows(FakeTokens(), signals, np.array([3, 1]), 1, "cpu", first)
    encode_windows(FakeTokens(), signals, np.array([3, 1]), 2, "cpu", second)
    tokens = np.load(first)
    np.testing.assert_array_equal(tokens, np.load(second))
    np.testing.assert_array_equal(tokens[:, :, 0, 0, 0], [[30, 31], [10, 11]])
    with pytest.raises(ValueError, match="Nonfinite selected"):
        encode_windows(FakeTokens(), signals, np.array([0]), 1, "cpu", tmp_path / "bad.npy")


def test_array_hash_and_shape_enforced(tmp_path):
    """Reject source corruption before encoding."""
    name = "2010_eeg.npy"
    np.save(tmp_path / name, np.zeros((1, 2, 64, 800), dtype=np.float32))
    metadata = {"array_sha256": {name: sha256(tmp_path / name)}}
    assert verified_array(tmp_path, name, metadata, (1, 2, 64, 800)).shape[0] == 1
    with pytest.raises(ValueError, match="shape/dtype"):
        verified_array(tmp_path, name, metadata, (2, 2, 64, 800))
    metadata["array_sha256"][name] = "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_array(tmp_path, name, metadata, (1, 2, 64, 800))


def test_smoke_integration_opens_only_selected_dyad(cache, tmp_path, monkeypatch):
    """Exercise artifacts and indexing with fake signals; any reserved-file access fails."""
    run = make_run(cache, tmp_path)
    _, _, selected, _ = feature_plan(cache, run, smoke=True)
    group = int(selected.group.iloc[0])
    rows = pd.read_csv(cache / "trials.csv")
    count = int(rows[(rows.group == group) & rows.retained].shape[0])
    eeg = np.zeros((count * 8, 2, 64, 800), dtype=np.float32)
    phase = np.zeros((count, 2, 4320), dtype=np.float32)
    opened = []

    def fake_array(root, name, metadata, shape):
        """Refuse arrays outside the selected development dyad."""
        assert name in (f"{group}_eeg.npy", f"{group}_phase.npy")
        opened.append(name)
        return eeg if name.endswith("eeg.npy") else phase

    class Extractor(FakeTokens):
        """Provide frozen normalization metadata for the integration path."""

        def __init__(self, checkpoint, device):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.backbone = nn.Module()
            self.encoder.backbone.fc_norm = nn.LayerNorm(200)
            self.max_reconstruction_error = 0.0

    monkeypatch.setattr("cl_behaviour.features.verified_array", fake_array)
    monkeypatch.setattr("cl_behaviour.features.FrozenTokens", Extractor)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("test")
    output = tmp_path / "features"
    result = extract_features(cache, run, output, checkpoint, smoke=True)
    assert len(opened) == 2 and result["selected_splits"] == ["train"]
    assert (output / "smoke.json").exists() and not (output / "features.json").exists()
    windows = pd.read_csv(output / "windows.csv")
    assert windows.movement_feature_stop.max() == 900
    assert windows.groupby("trial_id").movement_feature_index.nunique().eq(1).all()
    for name, digest in result["artifact_sha256"].items():
        assert sha256(output / name) == digest
