"""Regression checks for the original ForceGame solo-versus-trio task."""

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from labram_classification.__main__ import parser
from labram_classification.data import CHANNELS, discover, extract_windows, fold_datasets, prepare_cache


def interaction_epochs() -> mne.EpochsArray:
    """Create interleaved conditions with distinguishable signals and original epoch IDs."""
    names = ["T2P1", "T3P2n", "T1P1", "T3P1", "T1P2n", "rest"]
    codes = np.arange(1, len(names) + 1)
    signals = np.broadcast_to(codes[:, None, None] * 1e-6, (len(names), 64, 801)).copy()
    events = np.column_stack([codes * 1000, np.zeros(len(names), int), codes])
    return mne.EpochsArray(
        signals,
        mne.create_info(CHANNELS, 200, "eeg"),
        events=events,
        event_id=dict(zip(names, codes, strict=True)),
        tmin=-0.5,
        verbose=False,
    )


def test_interaction_filter_labels_and_provenance() -> None:
    """Match prefix labels regardless of feedback, preserving source epoch identity and units."""
    source = interaction_epochs()
    with pytest.raises(ValueError, match="801"):
        extract_windows(source, "fg", 301, "mixed", "solo-coordination", {}, fg_task="interaction")
    signals, rows = extract_windows(source, "fg", 301, "mixed", "solo-coordination", {}, "trim", "interaction")
    assert signals.shape == (4, 64, 800)
    assert rows.condition.tolist() == ["T3P2n", "T1P1", "T3P1", "T1P2n"]
    assert rows.target.tolist() == [1, 0, 1, 0]
    assert rows.source_epoch.tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(signals[:, 0, 0], [2, 3, 4, 5])
    assert len(source) == 6
    _, feedback = extract_windows(source, "fg", 301, "mixed", "solo-coordination", {}, "trim")
    assert feedback.target.tolist() == [1, 0, 1, 1, 0, 1]


def test_interaction_cache_and_triad_partitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publish an interaction cache consumable by both runners with unchanged triad folds."""
    directory = tmp_path / "sources"
    directory.mkdir()
    for group in range(301, 311):
        for member in "ABC":
            (directory / f"{group}{member}-epo.fif").touch()
    skipped = directory / "301A_rest-epo.fif"
    skipped.touch()
    source = interaction_epochs()

    def read_epochs(path: Path, **kwargs: object) -> mne.EpochsArray:
        """Supply actual MNE epochs without duplicating fixture recordings on disk."""
        return source[[0, 5]] if path == skipped else source

    monkeypatch.setattr(mne, "read_epochs", read_epochs)
    cache = tmp_path / "cache"
    prepare_cache(directory, cache, "fg", fg_endpoint="trim", fg_task="interaction")
    config = json.loads((cache / "cache.json").read_text())
    assert config["dataset"] == "fg"
    assert config["contrast"] == "interaction"
    assert config["n_windows"] == 120
    rows = pd.read_csv(cache / "windows.csv")
    assert str(skipped) not in set(rows.source)
    expected = discover(directory, "fg", "solo-coordination").groupby("group").fold.first()
    pd.testing.assert_series_equal(rows.groupby("group").fold.first(), expected)
    for fold in range(10):
        _, partitions = fold_datasets(cache, fold)
        assert [len(partition) for partition in partitions.values()] == [96, 12, 12]
        for partition in partitions.values():
            assert partition.metadata.groupby("group").participant.nunique().eq(3).all()
            assert set(partition.metadata.target) == {0, 1}
            signal, target, _ = partition[0]
            assert signal.shape == (64, 800)
            assert target in (0, 1)


def test_no_interaction_events_does_not_publish_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip ineligible files, but do not publish an entirely empty task as usable."""
    directory = tmp_path / "sources"
    directory.mkdir()
    for group in range(10):
        (directory / f"{group}A-epo.fif").touch()
    source = interaction_epochs()[[0, 5]]
    monkeypatch.setattr(mne, "read_epochs", lambda *args, **kwargs: source)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="No usable windows"):
        prepare_cache(directory, cache, "fg", fg_endpoint="trim", fg_task="interaction")
    assert not (cache / "cache.json").exists()


def test_fg_task_cli() -> None:
    """Keep feedback as the default and expose interaction through preparation only."""
    arguments = ["prepare", "--dataset", "fg", "--cache", "new-cache"]
    assert parser().parse_args(arguments).fg_task == "feedback"
    assert parser().parse_args(arguments + ["--fg-task", "interaction"]).fg_task == "interaction"
