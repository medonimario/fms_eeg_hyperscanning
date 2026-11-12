"""Tiny-fit readouts, memorization and training-only feature access checks."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from cl_behaviour.models import LinearHeads
from cl_behaviour.tiny_fit import center_scale, digest, fit_tiny_arm, load_tiny, readout, run_tiny_fit


def test_readout_axes_and_norm():
    """Retain channel/time ordering and apply pooling normalization after reduction."""
    tokens = torch.arange(2 * 64 * 4 * 200).reshape(2, 64, 4, 200).float()
    norm = dict(weight=np.ones(200, dtype=np.float32), bias=np.zeros(200, dtype=np.float32), eps=np.array(1e-6))
    for profile, shape in (("pooled", (2, 200)), ("temporal", (2, 800)), ("full", (2, 51200))):
        values = readout(tokens, norm, profile)
        assert values.shape == shape
        expected = tokens.mean((1, 2)) if profile == "pooled" else tokens.mean(1) if profile == "temporal" else tokens
        expected = torch.nn.functional.layer_norm(expected, (200,), eps=1e-6).flatten(1)
        torch.testing.assert_close(values, expected)
    with pytest.raises(ValueError, match="Degenerate"):
        center_scale(torch.ones(8, 20))


def test_true_and_shuffled_can_memorize():
    """The diagnostic should fit arbitrary assigned correspondences, not require shuffle failure."""
    generator = torch.Generator().manual_seed(7)
    eeg, _ = center_scale(torch.randn(8, 200, generator=generator))
    movement, _ = center_scale(torch.randn(8, 400, generator=generator))
    with torch.random.fork_rng():
        torch.manual_seed(42)
        initial = LinearHeads(200, 400, 32)
    for mapping in (torch.arange(8), torch.tensor([1, 2, 3, 4, 5, 6, 7, 0])):
        report, history, _, scores = fit_tiny_arm(initial, eeg, movement, mapping, steps=100)
        assert report["passed"]
        assert len(history) == 5 and scores.shape == (8, 8)


@pytest.fixture
def tiny_cache(tmp_path):
    """Create training/validation rows with poisoned held-out and partner feature slices."""
    features, run = tmp_path / "features", tmp_path / "split"
    features.mkdir()
    run.mkdir()
    trials, windows = [], []
    for index in range(9):
        trial = dict(
            trial_id=f"2010:closed1:{index}",
            group=2010,
            condition="closed1",
            condition_trial=index,
            split="train" if index < 8 else "validation",
            eligible=True,
            main_summary_eligible=True,
            cohort_fold=0,
        )
        trials.append(trial)
        windows.append(dict(trial, pair_id=f"trial:{index}:0", crop_start_s=0))
    source = pd.DataFrame(trials)
    source_windows = pd.DataFrame(windows)
    source.to_csv(run / "splits.csv", index=False)
    source_windows.to_csv(run / "windows.csv", index=False)
    protocol = dict(
        format_version=2,
        operation="split",
        profile="combined-closed",
        cache_sha256="prepared",
        manifest_sha256={name: digest(run / name) for name in ("splits.csv", "windows.csv")},
    )
    (run / "split.json").write_text(json.dumps(protocol))
    source["movement_feature_array"] = "2010_movement.npy"
    source["movement_feature_index"] = np.arange(9)
    source.to_csv(features / "trials.csv", index=False)
    source_windows["movement_feature_array"] = "2010_movement.npy"
    source_windows["movement_feature_index"] = np.arange(9)
    source_windows["token_array"] = "2010_tokens.npy"
    source_windows["token_array_index"] = np.arange(9)
    source_windows["movement_feature_start"] = 0
    source_windows["movement_feature_stop"] = 200
    source_windows.to_csv(features / "windows.csv", index=False)
    generator = np.random.default_rng(42)
    tokens = generator.normal(size=(9, 2, 64, 4, 200)).astype(np.float32)
    movement = generator.normal(size=(9, 2, 2, 900)).astype(np.float32)
    tokens[8] = np.nan
    movement[8] = np.nan
    tokens[:, 1] = np.nan
    movement[:, 1] = np.nan
    np.save(features / "2010_tokens.npy", tokens)
    np.save(features / "2010_movement.npy", movement)
    np.savez(
        features / "pooling_norm.npz",
        weight=np.ones(200, dtype=np.float32),
        bias=np.zeros(200, dtype=np.float32),
        eps=np.array(1e-6),
    )
    metadata = dict(
        format_version=1,
        operation="features",
        complete=True,
        backbone_frozen=True,
        members=["a", "b"],
        token_shape=[2, 64, 4, 200],
        movement_shape=[2, 2, 900],
        token_order="channel-major, time-minor",
        movement_components=["sin", "cos"],
        movement_sfreq=50,
        centered=False,
        statistics_fitted=False,
        profile="combined-closed",
        cache_sha256="prepared",
        split_sha256=digest(run / "split.json"),
        artifact_sha256={p.name: digest(p) for p in features.iterdir()},
    )
    (features / "features.json").write_text(json.dumps(metadata))
    return features, run


def test_training_only_loading_and_provenance(tiny_cache):
    """NaNs in validation and unselected member never reach the diagnostic."""
    features, run = tiny_cache
    tokens, movement, _, selected, _ = load_tiny(features, run)
    assert tokens.shape == (8, 64, 4, 200) and movement.shape == (8, 2, 200)
    assert selected.split.eq("train").all()
    assert torch.isfinite(tokens).all() and torch.isfinite(movement).all()
    with pytest.raises(ValueError, match="No permitted"):
        load_tiny(features, run, pairs=9)
    with (features / "windows.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_tiny(features, run)


def test_tiny_fit_outputs(tiny_cache, tmp_path):
    """Write an auditable successful two-arm result without held-out feature access."""
    features, run = tiny_cache
    output = tmp_path / "result"
    result = run_tiny_fit(features, run, output, steps=100)
    assert result["passed"] and not result["validation_features_materialized"]
    assert (output / "tiny_fit.json").is_file()
    for name, checksum in result["output_sha256"].items():
        assert digest(output / name) == checksum
    with pytest.raises(FileExistsError):
        run_tiny_fit(features, run, output, profiles=("pooled",), steps=1)
