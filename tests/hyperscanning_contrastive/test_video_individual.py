"""Independent movie selection, split inheritance and noise-combination checks."""

import json

import numpy as np
import pandas as pd
from test_video_labram import reference  # noqa: F401

from hyperscanning_contrastive.data import sha256
from hyperscanning_contrastive.video_baseline import run_video_baseline
from hyperscanning_contrastive.video_cnn import load_bank
from hyperscanning_contrastive.video_labram import load_reference


def test_new_movie_reference(reference, tmp_path):
    """Combine noise levels while excluding reserved viewers and retaining their original assignment."""
    cache, parent = reference
    people = pd.read_csv(parent / "participant_splits.csv")
    people["exclusion_reason"] = ""
    people.to_csv(cache / "participants.csv", index=False)
    old = pd.read_csv(cache / "windows.csv")
    old["noise_level"] = "none"
    rows, files = [], pd.read_csv(cache / "files.csv").to_dict("records")
    rng = np.random.default_rng(3)
    for person, noise in [("a", "low"), ("b", "high"), ("reserved", "low")]:
        path = f"{person}_video3.npy"
        digest = "must-not-open"
        if person != "reserved":
            np.save(cache / path, rng.normal(size=(64, 21600)).astype("float32"))
            digest = sha256(cache / path)
        files.append(dict(array_path=path, array_sha256=digest))
        for t in range(1, 27):
            rows.append(
                dict(
                    participant_id=person,
                    video_id=3,
                    condition="solo",
                    eligible=True,
                    noise_level=noise,
                    cache_start=t * 800,
                    cache_stop=(t + 1) * 800,
                    array_path=path,
                )
            )
    pd.concat([old, pd.DataFrame(rows)]).to_csv(cache / "windows.csv", index=False)
    pd.DataFrame(files).to_csv(cache / "files.csv", index=False)
    metadata = json.loads((cache / "cache.json").read_text())
    metadata["manifest_sha256"] = {p.name: sha256(p) for p in cache.glob("*.csv")}
    (cache / "cache.json").write_text(json.dumps(metadata))
    report = json.loads((parent / "baseline.json").read_text())
    report["cache_sha256"] = sha256(cache / "cache.json")
    (parent / "baseline.json").write_text(json.dumps(report))
    destination = tmp_path / "movie3"
    result = run_video_baseline(cache, destination, video_id=3, noise_level="combined", reference=parent)
    assert result["development_viewers"] == 2
    assert result["video_id"] == 3
    assert result["training_person_windows"] == 34
    _, galleries, windows = load_reference(cache, destination)
    assert set(windows.video_id) == {3}
    assert set(windows.noise_level) == {"low", "high"}
    bank, _ = load_bank(cache, destination, windows, standardize=False)
    assert len(bank["train"]) == 34
    assert all(p != "reserved" for split in bank.values() for p, _ in split)
    assert len(galleries) == 2
    saved = pd.read_csv(destination / "participant_splits.csv").set_index("participant_id")
    assert saved.loc["reserved", "split"] == "reserved"
