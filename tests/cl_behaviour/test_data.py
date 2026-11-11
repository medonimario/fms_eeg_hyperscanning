"""Whole-trial rejection and EEG–movement cache regression checks."""

import json
import pickle
from pathlib import Path

import mne
import numpy as np
import pytest

from cl_behaviour import data
from hyperscanning_contrastive.data import CHANNELS, CONDITIONS


def epochs(practice: int = 6) -> mne.EpochsArray:
    """Construct full experimental recordings with identifiable participant amplitudes."""
    codes = np.tile([1, 2, 3], 20 + practice // 3)
    events = np.column_stack((np.arange(len(codes)) * 6000, np.zeros(len(codes), int), codes))
    return mne.EpochsArray(
        np.full((len(codes), 64, 4000), 2e-6),
        mne.create_info(CHANNELS, 200, "eeg"),
        events=events,
        event_id=dict(zip(CONDITIONS, (1, 2, 3))),
        verbose=False,
    )


@pytest.mark.parametrize("pair,practice", [(2010, 6), (2030, 12), (2450, 6)])
def test_whole_trials(pair: int, practice: int) -> None:
    """Reject every section of affected trials and preserve eight exact offsets elsewhere."""
    first = epochs(practice)
    bad = {f"{pair}_{c}": [0, 4, 59] for c in CONDITIONS}
    windows, trials = data.aligned_plan(first, first.copy(), pair, bad)
    assert len(windows) == 3 * 17 * 8
    assert {r["condition_trial"] for r in trials if not r["retained"]} == {0, 1, 19}
    selected = [r for r in windows if r["condition"] == "open" and r["condition_trial"] == 2]
    assert [r["crop_start_s"] for r in selected] == list(range(0, 15, 2))
    assert selected[-1]["movement_stop_sample"] == 4320
    assert len({r["trial_id"] for r in selected}) == 1
    with pytest.raises(ValueError, match="Missing artifact"):
        data.aligned_plan(first, first, pair, {})


def test_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip one synthetic dyad through FIF discovery, phase correction and saved arrays."""
    monkeypatch.setattr(data, "PAIRS", (2450,))
    first = epochs()
    for member in "ab":
        first.save(tmp_path / f"pair2450{member}_ave_epo.fif", overwrite=True, verbose=False)
    phase = np.broadcast_to(np.arange(21)[None, None, :, None], (2, 3, 21, 4800)).copy().astype(float)
    phase_path, bad_path = tmp_path / "phase.pickle", tmp_path / "bad.pickle"
    phase_path.write_bytes(pickle.dumps([phase]))
    bad_path.write_bytes(pickle.dumps({f"2450_{c}": [0] for c in CONDITIONS}))
    output = tmp_path / "cache"
    result = data.prepare(tmp_path, phase_path, bad_path, output, True)
    assert result["retained_trials"] == 57
    assert result["paired_windows"] == 456
    eeg = np.load(output / "2450_eeg.npy")
    assert eeg.shape == (456, 2, 64, 800)
    np.testing.assert_allclose(eeg, 2, atol=1e-5)
    movement = np.load(output / "2450_phase.npy")
    assert movement.shape == (57, 2, 4320)
    assert movement[9, 0, 0] == 11
    assert json.loads((output / "cache.json").read_text())["split"].startswith("unassigned")
    with pytest.raises(FileExistsError):
        data.prepare(tmp_path, phase_path, bad_path, output, True)
