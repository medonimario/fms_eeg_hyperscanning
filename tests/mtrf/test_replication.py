"""Regression checks for trial selection, predictor direction, scores and notebook output layout."""

import pickle
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import zscore

from mTRF import eeg


def test_artifact_indices_and_extra_trial() -> None:
    """Lock the notebook's mixed indexing and pair 2450's missing middle trial."""
    assert eeg.retained_trials([0, 1, 20, 21, 59]) == [i for i in range(20) if i not in (1, 19)]
    tracks = np.broadcast_to(np.arange(21)[None, None, :, None], (2, 3, 21, 4800))
    phase = eeg.movement_phase(tracks, 2450)
    assert phase.shape == (2, 3, 20, 4320)
    np.testing.assert_array_equal(phase[1, 2, :, 0], np.delete(np.arange(21), 10))


def test_fit_and_collect(tmp_path: Path, monkeypatch) -> None:
    """Exercise FIF loading through real nested fits and compare one arm against literal notebook inputs."""
    rng = np.random.default_rng(41)
    root = tmp_path / "preprocessed"
    root.mkdir()
    pair = 2030
    phase = rng.normal(size=(2, 3, 20, 4800))
    event_ids = dict(zip(eeg.CONDITIONS, (11, 12, 13), strict=True))
    codes = np.tile([11, 12, 13], 24)
    events = np.column_stack((np.arange(len(codes)) * 5120, np.zeros_like(codes), codes))
    info = mne.create_info(["Cz"], 256, "eeg")
    for member in ("a", "b"):
        epochs = mne.EpochsArray(
            rng.normal(size=(len(codes), 1, 5120)), info, events=events, event_id=event_ids, verbose=False
        )
        epochs.save(root / f"pair2030{member}_ave_epo.fif", verbose=False)
    monkeypatch.setattr(eeg, "LAGS", (0, 0))
    monkeypatch.setattr(eeg, "LAMBDAS", np.array([0.1, 1.0]))
    bad = {f"{pair}_{c}": [5, 16] for c in eeg.CONDITIONS}
    scores, maps = eeg.fit_pair(tmp_path, pair, phase, bad, 1)
    keep = [i for i in range(20) if i not in (5, 16)]
    b, a = phase[1, 1, keep, 480:], phase[0, 1, keep, 480:]
    stim = list(np.stack([np.cos(b), np.sin(b), np.cos(a), np.sin(a)], axis=1).transpose(0, 2, 1))
    ref = mne.read_epochs(root / "pair2030b_ave_epo.fif", verbose=False).crop(tmin=2)[12:]["open"][keep].resample(240)
    resp = list(zscore(ref.get_data(), axis=-1).transpose(0, 2, 1))
    expected = eeg.nested_crossval(
        eeg.TRF(direction=1, preload=False), stim, resp, 240, 0, 0, eeg.LAMBDAS, k=5, seed=123, n_jobs=1, verbose=False
    )
    np.testing.assert_allclose(scores["open_SelfOther"][1], expected)
    assert len(scores) == 12 and np.shape(maps["open_self"]) == (2, 1, 1)
    monkeypatch.setattr(eeg, "PAIRS", (2020, 2030))
    for dyad in eeg.PAIRS:
        eeg.save_pickle(tmp_path / f"pair{dyad}.pickle", (scores, maps))
    eeg.collect(tmp_path)
    with (tmp_path / eeg.CV_FILE).open("rb") as handle:
        assert len(pickle.load(handle)["open_Full"]) == 4
    df = pd.read_csv(tmp_path / "delta_r2.csv")
    assert len(df) == 18 and set(df.pair) == {2030}
    full, reduced = (np.tanh(np.arctanh(scores[key][1][2]).mean()) for key in ("open_Full", "open_SelfCoordination"))
    np.testing.assert_allclose(
        df.query("member == 'b' and condition == 'Vision' and variable == 'other'").delta_r2, full - reduced
    )
