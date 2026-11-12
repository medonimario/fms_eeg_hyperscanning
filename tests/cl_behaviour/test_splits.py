"""Regression checks for metadata integrity and leak-free retrieval protocols."""

import json

import pandas as pd
import pytest

from cl_behaviour.splits import build_protocol, split_cache, validate_cache
from hyperscanning_contrastive.data import CHANNELS, CONDITIONS, sha256
from mTRF.common import PAIRS


@pytest.fixture
def cache(tmp_path):
    """Build a complete synthetic metadata cache without any signal arrays."""
    root = tmp_path / "cache"
    root.mkdir()
    trials, windows = [], []
    for group in PAIRS:
        phase_index = 0
        eeg_index = 0
        for condition_index, condition in enumerate(CONDITIONS):
            for repetition in range(20):
                keep = repetition != 5
                row = dict(
                    group=group,
                    condition=condition,
                    condition_trial=repetition,
                    trial_id=f"{group}:{condition}:{repetition}",
                    source_row=6 + 3 * repetition + condition_index,
                    practice_epochs_removed=6,
                    source_sfreq=256,
                    source_tmin=0,
                    retained=keep,
                    historical_mtrf_retained=keep,
                    retained_windows=8 if keep else 0,
                    main_summary_eligible=group != 2020,
                    split="unassigned",
                    movement_array=f"{group}_phase.npy",
                    movement_array_index=phase_index if keep else -1,
                )
                trials.append(row)
                if not keep:
                    continue
                phase_index += 1
                for start in range(0, 15, 2):
                    windows.append(
                        dict(
                            row,
                            pair_id=f"{row['trial_id']}:{start}",
                            crop_start_s=start,
                            crop_end_s=start + 4,
                            onset_s=start + 2,
                            end_s=start + 6,
                            start_sample=(start + 2) * 256,
                            stop_sample=(start + 6) * 256,
                            movement_start_sample=start * 240,
                            movement_stop_sample=(start + 4) * 240,
                            eeg_array=f"{group}_eeg.npy",
                            eeg_array_index=eeg_index,
                        )
                    )
                    eeg_index += 1
    trials = pd.DataFrame(trials)
    frames = {"trials.csv": trials, "windows.csv": pd.DataFrame(windows)}
    frames["census.csv"] = (
        trials.groupby(["group", "condition"])
        .agg(
            total_trials=("trial_id", "size"),
            retained_trials=("retained", "sum"),
            paired_windows=("retained_windows", "sum"),
        )
        .reset_index()
    )
    for name, frame in frames.items():
        frame.to_csv(root / name, index=False)
    metadata = dict(
        format_version=2,
        operation="prepare",
        dataset="circling-eeg-behaviour",
        artifact_policy="mtrf-notebook-exact",
        historical_exclusion_disagreements=0,
        members=["a", "b"],
        channels=CHANNELS,
        eeg_sfreq=200,
        movement_sfreq=240,
        eeg_units="microvolts",
        model_input_divisor=100,
        trial_interval_seconds=[2, 20],
        window_seconds=4,
        stride_seconds=2,
        retained_trials=int(trials.retained.sum()),
        paired_windows=len(windows),
        manifest_sha256={name: sha256(root / name) for name in frames},
        array_sha256={f"{group}_{kind}.npy": "a" * 64 for group in PAIRS for kind in ("phase", "eeg")},
    )
    (root / "cache.json").write_text(json.dumps(metadata))
    return root


def test_protocol_has_no_repetition_or_reserved_leakage(cache, tmp_path):
    """Keep reservations and paired repetitions intact, including discarded trials."""
    _, trials, _ = validate_cache(cache)
    folds = {group: i % 10 for i, group in enumerate(PAIRS)}
    splits, galleries = build_protocol(trials, folds, 2042)
    shuffled, shuffled_galleries = build_protocol(trials.sample(frac=1, random_state=7), folds, 2042)
    assert splits.set_index("trial_id").split.sort_index().equals(shuffled.set_index("trial_id").split.sort_index())
    pd.testing.assert_frame_equal(galleries, shuffled_galleries)
    assert splits.groupby("repetition_id").split.nunique().eq(1).all()
    assert not splits[splits.cohort_fold.isin([1, 2])].eligible.any()
    assert set(galleries.group).isdisjoint({g for g, f in folds.items() if f in (1, 2)})
    for gallery in galleries.itertuples():
        ids = json.loads(gallery.trial_ids)
        rows = splits.set_index("trial_id").loc[ids]
        assert rows.eligible.all() and rows.split.eq(gallery.split).all()
        assert rows.group.eq(gallery.group).all() and rows.condition.isin(["closed1", "closed2"]).all()
        assert rows.condition.tolist() == json.loads(gallery.trial_conditions)
        assert len(ids) == len(set(ids)) == gallery.candidate_count
        assert gallery.chance_top1 == 1 / len(ids)
    manifest = tmp_path / "cohort.json"
    manifest.write_text(json.dumps(dict(dataset="circling-local-project1", group_folds=folds)))
    output = tmp_path / "split"
    result = split_cache(cache, manifest, output)
    assert not result["signal_arrays_loaded"]
    assert (output / "split.json").exists()
    with pytest.raises(FileExistsError):
        split_cache(cache, manifest, output)


@pytest.mark.parametrize("column,value", [("movement_start_sample", 1), ("source_row", 999)])
def test_rejects_inconsistent_links_even_with_updated_hash(cache, column, value):
    """Detect timing or identity corruption independently of manifest hashes."""
    path = cache / "windows.csv"
    frame = pd.read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    metadata = json.loads((cache / "cache.json").read_text())
    metadata["manifest_sha256"]["windows.csv"] = sha256(path)
    (cache / "cache.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mismatch"):
        validate_cache(cache)


def test_rejects_hash_changes_and_incomplete_cohort(cache):
    """Refuse changed manifests and silently reconstructed reservations."""
    _, trials, _ = validate_cache(cache)
    with pytest.raises(ValueError, match="exactly"):
        build_protocol(trials, {2010: 0}, 2042)
    with (cache / "trials.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Hash mismatch"):
        validate_cache(cache)


def test_profiles_preserve_assignments_and_combined_gallery_union(cache):
    """Pooling changes candidates, never splits or original condition identities."""
    _, trials, _ = validate_cache(cache)
    folds = {group: i % 10 for i, group in enumerate(PAIRS)}
    combined, pooled = build_protocol(trials, folds, 2042)
    separate, individual = build_protocol(trials, folds, 2042, "separate-closed")
    opened, open_galleries = build_protocol(trials, folds, 2042, "open")
    assert combined.split.equals(separate.split) and combined.split.equals(opened.split)
    assert combined.eligible.equals(separate.eligible)
    assert open_galleries.condition.eq("open").all()
    for gallery in pooled.itertuples():
        matching = individual[
            (individual.group == gallery.group)
            & (individual.member == gallery.member)
            & (individual.split == gallery.split)
        ]
        union = {trial for ids in matching.trial_ids for trial in json.loads(ids)}
        assert set(json.loads(gallery.trial_ids)) == union
        assert len(matching) == 2
    assert not combined[combined.condition == "open"].eligible.any()
    clean = trials.copy()
    clean["retained"] = True
    _, full_galleries = build_protocol(clean, folds, 2042)
    assert full_galleries.groupby("split").candidate_count.unique().apply(list).to_dict() == {
        "test": [8],
        "train": [24],
        "validation": [8],
    }
    with pytest.raises(ValueError, match="Unknown gallery profile"):
        build_protocol(trials, folds, 2042, "unknown")
