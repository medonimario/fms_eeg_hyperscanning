"""Independent movie selection, split inheritance and noise-combination checks."""

import json

import numpy as np
import pandas as pd
import pytest
from test_video_labram import reference  # noqa: F401

from hyperscanning_contrastive.data import sha256
from hyperscanning_contrastive.video_baseline import run_video_baseline
from hyperscanning_contrastive.video_cnn import load_bank
from hyperscanning_contrastive.video_labram import load_reference


@pytest.mark.parametrize("cohort", ["solo", "together"])
def test_new_movie_reference(reference, tmp_path, cohort):
    """Combine noise levels while excluding reserved viewers and retaining their original assignment."""
    cache, parent = reference
    people = pd.read_csv(parent / "participant_splits.csv")
    people["exclusion_reason"] = ""
    people["condition"] = cohort
    people["independence_group"] = people.participant_id
    people.to_csv(parent / "participant_splits.csv", index=False)
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
                    condition=cohort,
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
    report["artifacts_sha256"]["participant_splits.csv"] = sha256(parent / "participant_splits.csv")
    (parent / "baseline.json").write_text(json.dumps(report))
    destination = tmp_path / "movie3"
    result = run_video_baseline(cache, destination, video_id=3, noise_level="combined", reference=parent, cohort=cohort)
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

    assert result["cohort"] == cohort
    assert result["development_independence_groups"] == 2
    if cohort == "together":
        source = pd.read_csv(cache / "windows.csv")
        source.loc[source.video_id == 3, "noise_level"] = "low"
        high = source[(source.video_id == 3) & (source.participant_id == "b")].copy()
        high["participant_id"] = "high_only"
        high["noise_level"] = "high"
        high["array_path"] = "missing_high_only.npy"
        source = pd.concat([source, high], ignore_index=True)
        extra_person = people[people.participant_id == "b"].copy()
        extra_person["participant_id"] = "high_only"
        extra_person["independence_group"] = "high_only"
        updated_people = pd.concat([people, extra_person], ignore_index=True)
        updated_people.to_csv(cache / "participants.csv", index=False)
        updated_people.to_csv(parent / "participant_splits.csv", index=False)
        metadata["manifest_sha256"]["participants.csv"] = sha256(cache / "participants.csv")
        report["artifacts_sha256"]["participant_splits.csv"] = sha256(parent / "participant_splits.csv")
        source.to_csv(cache / "windows.csv", index=False)
        metadata["manifest_sha256"]["windows.csv"] = sha256(cache / "windows.csv")
        (cache / "cache.json").write_text(json.dumps(metadata))
        report["cache_sha256"] = sha256(cache / "cache.json")
        (parent / "baseline.json").write_text(json.dumps(report))
        low = tmp_path / "low"
        run_video_baseline(cache, low, video_id=3, noise_level="low", reference=parent, cohort=cohort)
        _, _, selected = load_reference(cache, low)
        bank, _ = load_bank(cache, low, selected, standardize=False)
        assert len(bank["train"]) == 34

        assert "high_only" not in set(selected.index.get_level_values("participant_id"))
        saved_people = pd.read_csv(low / "participant_splits.csv")
        saved_people.loc[saved_people.participant_id.isin(["a", "reserved"]), "independence_group"] = "same_dyad"
        saved_people.to_csv(low / "participant_splits.csv", index=False)
        low_report = json.loads((low / "baseline.json").read_text())
        low_report["artifacts_sha256"]["participant_splits.csv"] = sha256(low / "participant_splits.csv")
        (low / "baseline.json").write_text(json.dumps(low_report))
        with pytest.raises(ValueError, match="independence group"):
            load_reference(cache, low)
