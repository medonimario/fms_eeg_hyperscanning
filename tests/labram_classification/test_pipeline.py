"""Regression checks for preprocessing, leakage, checkpoint loading, and updates."""

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from labram_classification.data import CHANNELS, discover, extract_windows, fold_datasets, sha256
from labram_classification.model import DEFAULT_CHECKPOINT, load_model
from labram_classification.training import BinaryLoss, FinetuneClassifier, learning_rates, parameter_groups


def epochs(data: np.ndarray, sfreq: float, tmin: float, labels: list[int], events: dict) -> mne.EpochsArray:
    """Build small EEG fixtures with real MNE preprocessing semantics."""
    info = mne.create_info(CHANNELS, sfreq, "eeg")
    event_array = np.column_stack([np.arange(len(labels)) * data.shape[-1], np.zeros(len(labels), int), labels])
    return mne.EpochsArray(data, info, events=event_array, event_id=events, tmin=tmin, verbose=False)


def test_mg_overlap_and_scaling() -> None:
    """Check retained overlap, pending reference, channel order, and microvolt units."""
    signal = np.arange(64 * 1200).reshape(1, 64, 1200) * 1e-9
    source = epochs(signal, 200, -1, [1], {"event": 1})
    data, meta = extract_windows(source, "mg", 1, "Solo", "solo-coordination", {})
    np.testing.assert_allclose(data[0], signal[0, :, :800] * 1e6, rtol=1e-6)
    np.testing.assert_array_equal(data[0, :, 400:], data[1, :, :400])
    assert meta.start_sample.tolist() == [0, 400]
    assert source.info["projs"] == []


def test_fg_endpoint_and_labels() -> None:
    """Make inclusive endpoint handling explicit rather than silently changing data."""
    source = epochs(np.ones((2, 64, 1001)) * 2e-6, 200, -0.5, [1, 2], {"feedback": 1, "feedbackn": 2})
    with pytest.raises(ValueError, match="801"):
        extract_windows(source, "fg", 301, "mixed", "solo-coordination", {})
    data, meta = extract_windows(source, "fg", 301, "mixed", "solo-coordination", {}, "trim")
    assert data.shape == (2, 64, 800)
    assert meta.target.tolist() == [1, 0]
    np.testing.assert_allclose(data, 2)


def test_circling_bad_segments_and_practice() -> None:
    """Exclude only overlapping windows, retaining the boundary at six seconds."""
    source = epochs(np.ones((8, 64, 4000)) * 3e-6, 200, 0, [1] * 7 + [2], {"open": 1, "closed1": 2})
    data, meta = extract_windows(source, "circling", 10, "mixed", "solo-coordination", {"10_open": [0]})
    assert set(meta.source_epoch) == {6, 7}
    assert meta[meta.condition == "open"].start_sample.tolist() == [1200, 1600, 2000, 2400, 2800]
    assert len(data) == 13
    np.testing.assert_allclose(data, 3)


def test_group_folds_and_cache(tmp_path: Path) -> None:
    """Both dyad members, both conditions, and overlapping windows stay together."""
    for group in range(1, 11):
        for member in (1, 2):
            for condition in ("Solo", "Coordination", "Spontaneous"):
                (tmp_path / f"s_{member}{group:02d}_{condition}.set").touch()
    files = discover(tmp_path, "mg", "solo-coordination")
    assert len(files) == 40
    files["target"] = files.condition.map({"Solo": 0, "Coordination": 1})
    files["array"] = "data.npy"
    files["array_index"] = 0
    np.save(tmp_path / "data.npy", np.ones((1, 64, 800), np.float32) * 100)
    files.to_csv(tmp_path / "windows.csv", index=False)
    (tmp_path / "cache.json").write_text(
        json.dumps(dict(format_version=1, channels=CHANNELS, metadata_sha256=sha256(tmp_path / "windows.csv")))
    )
    for fold in range(10):
        _, partitions = fold_datasets(tmp_path, fold)
        assert [len(value) for value in partitions.values()] == [32, 4, 4]
        np.testing.assert_allclose(partitions["train"][0][0], 1)


def test_schedule() -> None:
    """Check the original warmup endpoint and cosine denominator."""
    expected = np.r_[np.linspace(0, 0.01, 4), 0.001 + 0.0045 * (1 + np.cos(np.pi * np.arange(4) / 4))]
    np.testing.assert_allclose(learning_rates(0.01, 0.001, 4, 2, 2), expected)


def test_accumulation_matches_large_batch() -> None:
    """Two microbatches produce the same AdamW update as their combined batch."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 1), nn.Flatten(0))
    reference = nn.Sequential(nn.Linear(3, 1), nn.Flatten(0))
    reference.load_state_dict(model.state_dict())
    x, y = torch.randn(4, 3), torch.tensor([0, 1, 1, 0])
    net = FinetuneClassifier(
        model,
        criterion=BinaryLoss,
        classes=[0, 1],
        lr=0.001,
        layer_decay=1,
        weight_decay=0.05,
        update_freq=2,
        max_epochs=1,
        warmup_epochs=0,
        batch_size=2,
        train_split=None,
        callbacks="disable",
        iterator_train__shuffle=False,
        verbose=0,
        device="cpu",
    )
    net.fit(TensorDataset(x, y), y=None)
    optimizer = torch.optim.AdamW(parameter_groups(reference, 0.001, 1, 0.05), lr=0.001, eps=1e-8)
    BinaryLoss()(reference(x), y).backward()
    optimizer.step()
    for actual, expected in zip(net.module_.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="Local pretrained checkpoint unavailable")
def test_local_checkpoint_forward_backward() -> None:
    """Strictly load the actual downloaded checkpoint and exercise every trainable parameter."""
    torch.set_num_threads(2)
    model = load_model(drop_path=0)
    torch.testing.assert_close(model.backbone.fc_norm.weight, torch.ones(200))
    model.eval()
    logits = model(torch.randn(2, 64, 800))
    assert logits.shape == (2,)
    BinaryLoss()(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("initialization", ["pretrained", "random"])
def test_real_model_training_and_exports(tmp_path: Path, initialization: str) -> None:
    """Exercise EEGClassifier selection, checkpoint restore, and ordered predictions end to end."""
    from labram_classification.runner import run_fold

    if initialization == "pretrained" and not DEFAULT_CHECKPOINT.exists():
        pytest.skip("Local pretrained checkpoint unavailable")
    torch.set_num_threads(2)
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.RandomState(0)
    np.save(cache / "signals.npy", rng.normal(size=(2, 64, 800)).astype(np.float32) * 20)
    rows = [
        dict(
            group=fold,
            participant=str(fold),
            fold=fold,
            target=target,
            array="signals.npy",
            array_index=target,
            source="synthetic",
        )
        for fold in range(10)
        for target in (0, 1)
    ]
    pd.DataFrame(rows).to_csv(cache / "windows.csv", index=False)
    (cache / "cache.json").write_text(
        json.dumps(
            dict(format_version=1, channels=CHANNELS, dataset="mg", metadata_sha256=sha256(cache / "windows.csv"))
        )
    )
    options = dict(
        initialization=initialization,
        lr=1e-4,
        min_lr=1e-6,
        epochs=1,
        warmup_epochs=0,
        batch_size=2,
        update_freq=2,
        layer_decay=0.65,
        weight_decay=0.05,
        amp=False,
        device="cpu",
        num_workers=0,
        seed=0,
        drop_path=0,
        smoke_windows_per_class=2,
    )
    output = tmp_path / "fold"
    checkpoint = DEFAULT_CHECKPOINT if initialization == "pretrained" else tmp_path / "absent.pth"
    result = run_fold(cache, output, checkpoint, 0, options)
    config = json.loads((output / "config.json").read_text())
    assert config["initialization"] == initialization
    assert (config["checkpoint_sha256"] is None) == (initialization == "random")
    assert result["best_epoch"] == 1
    assert result["smoke"]
    assert (output / "best_model.pt").is_file()
    predictions = pd.read_csv(output / "test_predictions.csv")
    assert predictions.target.tolist() == [0, 1]
    assert predictions.probability_class_1.between(0, 1).all()
