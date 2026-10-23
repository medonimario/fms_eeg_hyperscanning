"""Stage0 video identity, timing, QC and signal-cache contracts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import resample_poly

from hyperscanning_contrastive.data import CHANNELS, sha256
from hyperscanning_contrastive.video_data import load_recording, noise_level, prepare_videos


def recording(root: Path, subject: str, video: int, samples: int = 8192, spike: bool = False) -> Path:
    """Write trusted synthetic NPZ signals with shuffled channel order and original-rate metadata."""
    condition = "solo" if subject.startswith("solo") else "together"
    folder = root / f"{condition}_condition" / subject
    folder.mkdir(parents=True, exist_ok=True)
    (root / "solo_condition").mkdir(exist_ok=True)
    (root / "together_condition").mkdir(exist_ok=True)
    rng = np.random.default_rng(sum(map(ord, subject)) + video)
    data = rng.normal(0, 5e-6, (samples, 64))
    if spike:
        data[2200, 0] = 0.001
    path = folder / f"video{video}.npz"
    np.savez(
        path,
        data=data,
        ch_names=np.array(CHANNELS[::-1], dtype=object),
        sfreq=256.0,
        video_number=video,
        video_start_code=3,
        video_end_code=4,
        start_sample=4096,
        end_sample=4096 + samples * 8,
    )
    return path


def test_audit_prepare(tmp_path: Path) -> None:
    """Windows stay aligned, artifacts remain local, excluded partners do not remove good viewers."""
    root = tmp_path / "source"
    p = recording(root, "solo1_A_1", 3, spike=True)
    recording(root, "solo2_B_2", 3, samples=8192 - 32)
    recording(root, "solo11_A_1", 3)
    recording(root, "pair4_together_A_4_p1", 3)
    recording(root, "pair4_together_A_4_p2", 3)
    audit = prepare_videos(root, tmp_path / "audit")
    cache = prepare_videos(root, tmp_path / "cache", prepare=True)
    assert audit["usable_person_windows"] == cache["usable_person_windows"]
    assert audit["arrays_written"] == 0
    assert cache["arrays_written"] == 3
    assert cache["included_participants"] == 3
    assert not cache["evaluation_ready"]
    windows = pd.read_csv(tmp_path / "cache/windows.csv", keep_default_na=False)
    bad = windows[(windows.participant_id == "solo1_A_1") & (windows.native_start == 2048)]
    assert not bad.eligible.any()
    assert bad.reason.str.contains("amplitude").all()
    good_partner = windows[(windows.participant_id == "pair4_together_A_4_p1") & windows.eligible]
    assert len(good_partner) > 0
    assert (windows.loc[windows.eligible, "end_seconds"] <= (8192 - 32) / 256 - 4).all()
    assert set(windows.split) == {"unassigned"}
    files = pd.read_csv(tmp_path / "cache/files.csv", keep_default_na=False)
    row = files[files.participant_id == "solo1_A_1"].iloc[0]
    signal = np.load(tmp_path / "cache" / row.array_path)
    original, _ = load_recording(p)
    np.testing.assert_allclose(signal, resample_poly(original, 25, 32, axis=1), rtol=1e-6, atol=1e-5)
    with np.load(p, allow_pickle=True) as raw:
        np.testing.assert_allclose(original[0], raw["data"][:, -1] * 1e6)
    assert sha256(tmp_path / "cache" / row.array_path) == row.array_sha256
    eligible = good_partner.iloc[0]
    assert eligible.cache_stop - eligible.cache_start == 800
    assert eligible.native_stop - eligible.native_start == 1024
    assert not (tmp_path / "audit/signals").exists()
    with pytest.raises(FileExistsError):
        prepare_videos(root, tmp_path / "audit")


def test_metadata_rejection_and_noise(tmp_path: Path) -> None:
    """Mislabelled movies fail and noise mapping reverses by counterbalance."""
    assert noise_level(0, "A") == "none"
    assert noise_level(3, "A") == noise_level(8, "B") == "low"
    assert noise_level(3, "B") == noise_level(8, "A") == "high"
    p = recording(tmp_path, "solo1_A_1", 3)
    p.rename(p.with_name("video4.npz"))
    with pytest.raises(ValueError, match="identity"):
        load_recording(p.with_name("video4.npz"))


def test_nonfinite_and_duplicate(tmp_path: Path) -> None:
    """Nonfinite EEG cannot enter a continuous resampler; duplicate signals are explicit."""
    root = tmp_path / "source"
    p = recording(root, "solo1_A_1", 0)
    q = recording(root, "solo2_A_2", 0)
    q.write_bytes(p.read_bytes())
    bad = recording(root, "solo3_A_3", 0)
    with np.load(bad, allow_pickle=True) as f:
        values = dict(f)
    values["data"][10, 0] = np.nan
    np.savez(bad, **values)
    report = prepare_videos(root, tmp_path / "cache", prepare=True)
    assert report["arrays_written"] == 1
    assert "duplicate_signal" in report["window_reasons"]
    assert "nonfinite_recording" in report["window_reasons"]
    saved = json.loads((tmp_path / "cache/cache.json").read_text())
    assert saved["manifest_sha256"]["windows.csv"] == sha256(tmp_path / "cache/windows.csv")
