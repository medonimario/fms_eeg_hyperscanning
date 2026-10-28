"""Regression controls for the held-time video pilot."""

import json

import numpy as np
import pandas as pd
import pytest

from hyperscanning_contrastive.data import CHANNELS, sha256
from hyperscanning_contrastive.video_baseline import fit_corrca, run_video_baseline, score_gallery, waveform_features


def test_shared_signal_retrieval():
    """Shared filters recover an aligned component on entirely new waveforms."""
    rng = np.random.default_rng(6)
    mixing = rng.normal(size=(8, 1))
    bank = {str(p): {} for p in range(8)}
    for t in range(12):
        shared = mixing @ rng.normal(size=(1, 400))
        for values in bank.values():
            values[t] = shared + rng.normal(size=(8, 400))
    filters, eigenvalues, _ = fit_corrca(bank, 1, 0.1)
    assert eigenvalues[0] > 0.5
    a, b = [], []
    for _ in range(4):
        shared = mixing @ rng.normal(size=(1, 400))
        a.append(waveform_features(shared + rng.normal(size=(8, 400)), filters))
        b.append(waveform_features(shared + rng.normal(size=(8, 400)), filters))
    assert score_gallery(np.array(a), np.array(b))["top1"] == 1
    assert score_gallery(np.array(a), np.roll(b, 1, axis=0))["top1"] == 0


def test_pilot_integrity_and_splits(tmp_path):
    """Check portable cache loading, disjoint times, reserved IDs and hash enforcement."""
    root = tmp_path / "cache"
    root.mkdir()
    rng = np.random.default_rng(3)
    people, windows, files = [], [], []
    for p in range(6):
        person = f"solo{p}_A_{p}"
        people.append(
            dict(
                participant_id=person,
                independence_group=f"solo{p}",
                condition="solo",
                counterbalance="A",
                exclusion_reason="",
            )
        )
        path = f"{person}.npy"
        np.save(root / path, rng.normal(size=(64, 800 * 27)).astype("float32"))
        files.append(dict(array_path=path, array_sha256=sha256(root / path)))
        for t in range(1, 27):
            windows.append(
                dict(
                    participant_id=person,
                    condition="solo",
                    video_id=0,
                    noise_level="none",
                    eligible=True,
                    cache_start=t * 800,
                    cache_stop=(t + 1) * 800,
                    array_path=path,
                )
            )
    for name, records in [("participants", people), ("files", files), ("windows", windows)]:
        pd.DataFrame(records).to_csv(root / f"{name}.csv", index=False)
    metadata = dict(
        dataset="shared-stimulus-video",
        operation="prepare",
        channels=CHANNELS,
        sfreq=200,
        units="microvolts",
        manifest_sha256={p.name: sha256(p) for p in root.glob("*.csv")},
    )
    (root / "cache.json").write_text(json.dumps(metadata))
    output = tmp_path / "run"
    report = run_video_baseline(root, output)
    assert report["development_viewers"] == 4
    assert not report["reserved_people_evaluated"]
    splits = pd.read_csv(output / "window_splits.csv")
    assert (
        splits[splits.time_split == "train"].cache_stop.max() + 800
        <= splits[splits.time_split == "validation"].cache_start.min()
    )
    galleries = pd.read_csv(output / "galleries.csv")
    reserved = pd.read_csv(output / "participant_splits.csv").query("split == 'reserved'").participant_id
    assert not set(galleries["first"]) & set(reserved)
    assert (output / "baseline.png").is_file()
    with (root / "windows.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_video_baseline(root, tmp_path / "bad")


def test_ties_and_independent_signals():
    """Uninformative ties score at chance and independent EEG lacks strong CorrCA eigenvalues."""
    metrics = score_gallery(np.zeros((4, 8)), np.zeros((4, 8)))
    assert metrics["top1"] == 0.25
    assert metrics["mrr"] == pytest.approx(sum(1 / np.arange(1, 5)) / 4)
    rng = np.random.default_rng(12)
    bank = {str(p): {t: rng.normal(size=(8, 400)) for t in range(12)} for p in range(8)}
    _, eigenvalues, _ = fit_corrca(bank, 3, 0.1)
    assert eigenvalues[0] < 0.05
