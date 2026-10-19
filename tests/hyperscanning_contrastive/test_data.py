"""Regression checks for local Circling pairing, artifact exclusion, and caches."""

from __future__ import annotations

import itertools
import json
import pickle
from pathlib import Path

import mne
import numpy as np
import pytest

from hyperscanning_contrastive.data import (
    CHANNELS,
    clean_windows,
    discover,
    extract_pair,
    plan_pair,
    prepare,
    read_bad_segments,
    sha256,
)


def make_epochs(value: float = 2e-6, practice: int = 6, sfreq: int = 100) -> mne.EpochsArray:
    """Build two practice cycles and one experimental cycle on a shared acquisition clock."""
    codes = np.tile([11, 12, 13], practice // 3 + 1)
    events = np.column_stack((np.arange(len(codes)) * 30 * sfreq, np.zeros(len(codes), int), codes))
    data = np.full((len(codes), 64, 20 * sfreq), value)
    return mne.EpochsArray(
        data,
        mne.create_info(CHANNELS, sfreq, "eeg"),
        events=events,
        event_id=dict(closed1=11, open=12, closed2=13),
        verbose=False,
    )


@pytest.mark.parametrize(
    ("bad", "expected"),
    [
        ([], [(0, 4), (4, 8), (8, 12), (12, 16)]),
        ([0], [(6, 10), (10, 14), (14, 18)]),
        ([1], [(0, 4), (12, 16)]),
        ([2], [(0, 4), (4, 8), (8, 12)]),
        ([0, 1, 2], []),
    ],
)
def test_clean_intervals(bad: list[int], expected: list[tuple[int, int]]) -> None:
    """Keep maximal clean windows while treating touching boundaries as non-overlap."""
    assert clean_windows(bad) == expected


def test_all_artifact_patterns() -> None:
    """Verify optimal packing, disjoint windows, and no artifact intersection for every section mask."""
    for mask in itertools.product((False, True), repeat=3):
        sections = [i for i, marked in enumerate(mask) if marked]
        windows = clean_windows(sections)
        assert all(b <= c for (_, b), (c, _) in zip(windows, windows[1:]))
        assert all(not (a < 6 * (section + 1) and b > 6 * section) for a, b in windows for section in sections)
        clean_runs = [len(list(run)) * 6 for is_bad, run in itertools.groupby(mask) if not is_bad]
        assert len(windows) == sum(length // 4 for length in clean_runs)
    assert clean_windows([0], "fixed-grid") == [(8, 12), (12, 16)]


def test_pair_mapping_and_symmetric_exclusion() -> None:
    """Resolve six-second sections after practice removal and keep the same intervals for both people."""
    first, second = make_epochs(), make_epochs(7e-6)
    rows, trials = plan_pair(first, second, 2010, {"2010_open": [0]})
    active = [row for row in rows if row["condition"] == "open"]
    assert [row["onset_s"] for row in active] == [8, 12, 16]
    assert {row["source_row"] for row in active} == {7}
    assert {row["source_epoch"] for row in active} == {7}
    assert len(trials) == 3
    data = extract_pair(first, second, active)
    assert data.shape == (3, 2, 64, 800)
    assert data.dtype == np.float32
    np.testing.assert_allclose(data[:, 0], 2, atol=1e-5)
    np.testing.assert_allclose(data[:, 1], 7, atol=1e-5)


def test_pair2030_practice() -> None:
    """Preserve the local special case of four practice cycles for pair2030."""
    epochs = make_epochs(practice=12)
    rows, _ = plan_pair(epochs, epochs.copy(), 2030, {}, conditions=("open",))
    assert {row["source_row"] for row in rows} == {13}
    assert {row["practice_epochs_removed"] for row in rows} == {12}


def test_artifact_index_maps_to_second_experimental_trial() -> None:
    """Index three excludes the first section of the second condition trial, not a practice epoch."""
    epochs = make_epochs(practice=9)
    rows, trials = plan_pair(epochs, epochs.copy(), 2010, {"2010_open": [3]}, conditions=("open",))
    assert [trial["retained_windows"] for trial in trials] == [4, 3]
    assert [row["source_row"] for row in rows] == [7] * 4 + [10] * 3
    assert [row["onset_s"] for row in rows if row["condition_trial"] == 1] == [8, 12, 16]


def test_sample_slices_and_channel_order() -> None:
    """Check exact crop-origin samples and channel ordering using a time/channel-coded signal."""
    epochs = make_epochs(sfreq=200)
    epochs._data[:] = np.arange(4000)[None, None, :] * 1e-6 + np.arange(64)[None, :, None] * 0.01
    reordered = epochs.copy().reorder_channels(list(reversed(CHANNELS)))
    rows, _ = plan_pair(epochs, reordered, 2010, {"2010_open": [0]}, conditions=("open",))
    signal = extract_pair(epochs, reordered, rows)
    np.testing.assert_array_equal(signal[:, 0], signal[:, 1])
    np.testing.assert_allclose(signal[0, 0, 0], np.arange(1600, 2400), atol=0.001)
    np.testing.assert_allclose(signal[-1, 0, 0], np.arange(3200, 4000), atol=0.001)


@pytest.mark.parametrize("mutation", ["event", "drop", "time", "channel"])
def test_ambiguous_alignment_fails(mutation: str) -> None:
    """Never pair independently changed recordings by surviving row order."""
    first, second = make_epochs(), make_epochs()
    if mutation == "event":
        second.events[7, 0] += 1
    elif mutation == "drop":
        second.drop([6], verbose=False)
    elif mutation == "time":
        second.shift_time(0.01, relative=True)
    else:
        second.drop_channels([CHANNELS[0]])
    with pytest.raises(ValueError):
        plan_pair(first, second, 2010, {})


def test_matching_dropped_epochs_still_fail() -> None:
    """Even matching exclusions need original dictionary-to-trial provenance before reuse."""
    epochs = make_epochs()
    epochs.drop([6], verbose=False)
    with pytest.raises(ValueError, match="ambiguous"):
        plan_pair(epochs, epochs.copy(), 2010, {})


@pytest.mark.parametrize("offset,stride", [(1, 1), (2, 3)])
def test_ignored_event_gaps_preserve_artifact_mapping(tmp_path: Path, offset: int, stride: int) -> None:
    """Keep condition-local artifact indices across FIF roundtrips with ignored source events."""
    epochs = make_epochs()
    epochs.selection = offset + np.arange(len(epochs)) * stride
    selected = set(epochs.selection)
    epochs.drop_log = tuple(() if i in selected else ("IGNORED",) for i in range(int(epochs.selection[-1]) + 2))
    path = tmp_path / "ignored-epo.fif"
    epochs.save(path, verbose=False)
    restored = mne.read_epochs(path, preload=False, verbose=False)
    rows, _ = plan_pair(restored, restored.copy(), 2010, {"2010_open": [0]}, conditions=("open",))
    assert [row["onset_s"] for row in rows] == [8, 12, 16]
    assert {row["source_row"] for row in rows} == {7}
    assert {row["source_epoch"] for row in rows} == {offset + 7 * stride}
    assert {row["condition_trial"] for row in rows} == {0}


@pytest.mark.parametrize("reason", [(), ("USER",), ("BAD_blink",), ("IGNORED", "USER")])
def test_unresolved_gap_provenance_fails(reason: tuple[str, ...]) -> None:
    """Reject unexplained gaps and real drops rather than shifting artifact indices."""
    epochs = make_epochs()
    epochs.selection = np.arange(len(epochs)) + 1
    epochs.drop_log = (reason,) + tuple(() for _ in epochs.selection)
    with pytest.raises(ValueError, match="Drop-log examples"):
        plan_pair(epochs, epochs.copy(), 2010, {})


def test_out_of_range_artifact_fails() -> None:
    """Reject a dictionary whose trial numbering is incompatible with the recordings."""
    epochs = make_epochs()
    with pytest.raises(ValueError, match="exceeds"):
        plan_pair(epochs, epochs.copy(), 2010, {"2010_open": [3]})


@pytest.mark.parametrize("value", [[-1], [0.5], [True], "0"])
def test_bad_dictionary_schema(tmp_path: Path, value: object) -> None:
    """Reject invalid artifact indices before processing signals."""
    path = tmp_path / "bad.pickle"
    path.write_bytes(pickle.dumps({"2010_open": value}))
    with pytest.raises(ValueError):
        read_bad_segments(path)


def test_missing_pair_fails(tmp_path: Path) -> None:
    """Require both members of every discovered dyad."""
    (tmp_path / "pair2010a_ave_epo.fif").touch()
    with pytest.raises(ValueError, match="exactly two"):
        discover(tmp_path)


def test_audit_and_cache_end_to_end(tmp_path: Path) -> None:
    """Audit saved FIF files and verify paired cache shape, provenance, units, and completion markers."""
    source = tmp_path / "preprocessed"
    source.mkdir()
    make_epochs().save(source / "pair2010a_ave_epo.fif", verbose=False)
    make_epochs(7e-6).save(source / "pair2010b_ave_epo.fif", verbose=False)
    bad_path = tmp_path / "bad.pickle"
    bad_path.write_bytes(pickle.dumps({"2010_open": [0], "2010_closed2": [0, 1, 2]}))
    audit = prepare(source, tmp_path / "audit", bad_path)
    assert audit["paired_window_count"] == 7
    assert not list((tmp_path / "audit").glob("*.npy"))
    assert not (tmp_path / "audit" / "cache.json").exists()
    cache_dir = tmp_path / "cache"
    summary = prepare(source, cache_dir, bad_path, write_signals=True)
    assert summary["artifact_segment_seconds"] == 6
    assert summary["missing_artifact_keys"] == ["2010_closed1"]
    assert not summary["evaluation_ready"]
    assert summary["array_sha256"]["pair_2010.npy"] == sha256(cache_dir / "pair_2010.npy")
    signal = np.load(cache_dir / "pair_2010.npy", allow_pickle=False)
    assert signal.shape == (7, 2, 64, 800)
    np.testing.assert_allclose(signal[:, 0], 2, atol=1e-5)
    np.testing.assert_allclose(signal[:, 1], 7, atol=1e-5)
    assert json.loads((cache_dir / "cache.json").read_text())["operation"] == "prepare"
    with pytest.raises(FileExistsError):
        prepare(source, cache_dir, bad_path)


def test_nonfinite_signal_no_completion(tmp_path: Path) -> None:
    """Reject corrupt signal data and leave no successful cache marker."""
    source = tmp_path / "preprocessed"
    source.mkdir()
    first = make_epochs()
    first._data[6, 0, 200] = np.nan
    first.save(source / "pair2010a_ave_epo.fif", verbose=False)
    make_epochs().save(source / "pair2010b_ave_epo.fif", verbose=False)
    bad_path = tmp_path / "bad.pickle"
    bad_path.write_bytes(pickle.dumps({}))
    with pytest.raises(ValueError, match="Invalid signal"):
        prepare(source, tmp_path / "cache", bad_path, write_signals=True)
    assert not (tmp_path / "cache" / "cache.json").exists()
