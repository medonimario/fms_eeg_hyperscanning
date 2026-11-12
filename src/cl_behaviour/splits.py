"""Validate prepared metadata and freeze repetition splits and retrieval galleries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hyperscanning_contrastive.data import CHANNELS, CONDITIONS, sha256
from mTRF.common import PAIRS


def require(condition: bool, message: str) -> None:
    """Reject an inconsistent input with a descriptive error."""
    if not condition:
        raise ValueError(message)


def validate_cache(cache: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Validate hashed metadata without opening signal arrays or trusted pickles."""
    metadata = json.loads((cache / "cache.json").read_text())
    expected = dict(
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
    )
    for key, value in expected.items():
        require(metadata.get(key) == value, f"Invalid cache contract: {key}")
    for name in ("trials.csv", "windows.csv", "census.csv"):
        require(sha256(cache / name) == metadata["manifest_sha256"][name], f"Hash mismatch: {name}")
    trials = pd.read_csv(cache / "trials.csv", keep_default_na=False)
    windows = pd.read_csv(cache / "windows.csv", keep_default_na=False)
    require(set(trials.group) == set(PAIRS), "Expected all 52 dyads")
    require(not trials.trial_id.duplicated().any(), "Duplicate trial identities")
    require(not windows.pair_id.duplicated().any(), "Duplicate window identities")
    require(set(trials.condition) == set(CONDITIONS), "Unexpected conditions")
    require((trials.retained == trials.historical_mtrf_retained).all(), "mTRF exclusion disagreement")
    require((trials.main_summary_eligible == (trials.group != 2020)).all(), "Pair 2020 summary policy")
    require((trials.split == "unassigned").all(), "Cache already assigned")
    for (_, condition), rows in trials.groupby(["group", "condition"]):
        rows = rows.sort_values("condition_trial")
        require(rows.condition_trial.tolist() == list(range(20)), "Missing or duplicate repetitions")
        expected_rows = rows.practice_epochs_removed + rows.condition_trial * 3 + CONDITIONS.index(condition)
        require((rows.source_row == expected_rows).all(), "Cannot establish cross-condition repetition identity")
    require((trials.retained_windows == trials.retained.astype(int) * 8).all(), "Invalid window counts")
    retained = trials[trials.retained]
    require(len(retained) == metadata["retained_trials"], "Retained count mismatch")
    require(len(windows) == metadata["paired_windows"] == len(retained) * 8, "Window total mismatch")
    require(set(windows.trial_id) == set(retained.trial_id), "Windows reference excluded/missing trials")
    parent = trials.set_index("trial_id", drop=False).loc[windows.trial_id].reset_index(drop=True)
    for column in trials.columns:
        require((windows[column].reset_index(drop=True) == parent[column]).all(), f"Window parent mismatch: {column}")
    for _, rows in windows.groupby("trial_id"):
        require(sorted(rows.crop_start_s) == list(range(0, 15, 2)), "Incomplete window offsets")
    for column, values in {
        "crop_end_s": windows.crop_start_s + 4,
        "onset_s": windows.crop_start_s + 2,
        "end_s": windows.crop_start_s + 6,
        "movement_start_sample": windows.crop_start_s * 240,
        "movement_stop_sample": (windows.crop_start_s + 4) * 240,
        "start_sample": ((windows.crop_start_s + 2 - windows.source_tmin) * windows.source_sfreq).round(),
        "stop_sample": ((windows.crop_start_s + 6 - windows.source_tmin) * windows.source_sfreq).round(),
    }.items():
        require((windows[column] == values).all(), f"Timing mismatch: {column}")
    for group, rows in retained.groupby("group"):
        eeg_rows = windows[windows.group == group]
        require(sorted(rows.movement_array_index) == list(range(len(rows))), "Phase array indices")
        require(sorted(eeg_rows.eeg_array_index) == list(range(len(eeg_rows))), "EEG array indices")
        require((rows.movement_array == f"{group}_phase.npy").all(), "Phase filename mismatch")
        require((eeg_rows.eeg_array == f"{group}_eeg.npy").all(), "EEG filename mismatch")
        for name in (f"{group}_phase.npy", f"{group}_eeg.npy"):
            require(len(metadata["array_sha256"].get(name, "")) == 64, "Missing array digest")
    census = (
        trials.groupby(["group", "condition"])
        .agg(
            total_trials=("trial_id", "size"),
            retained_trials=("retained", "sum"),
            paired_windows=("retained_windows", "sum"),
        )
        .reset_index()
    )
    pd.testing.assert_frame_equal(census, pd.read_csv(cache / "census.csv"), check_dtype=False)
    return metadata, trials, windows


PROFILES = ("combined-closed", "separate-closed", "open")


def build_protocol(
    trials: pd.DataFrame, folds: dict[int, int], seed: int, profile: str = "combined-closed"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign paired repetitions and emit profile-specific same-person galleries."""
    require(profile in PROFILES, f"Unknown gallery profile: {profile}")
    conditions = ("open",) if profile == "open" else ("closed1", "closed2")
    require(set(folds) == set(trials.group), "Cohort manifest must cover exactly the cache dyads")
    require(set(folds.values()) == set(range(10)), "Expected original ten-fold cohort")
    result = trials.copy()
    result["cohort_fold"] = result.group.map(folds)
    result["repetition_id"] = result.group.astype(str) + ":" + result.condition_trial.astype(str)
    assignments = {}
    for group in sorted(folds):
        order = np.random.default_rng(np.random.SeedSequence([seed, group])).permutation(20)
        for index, repetition in enumerate(order):
            split = "validation" if index < 4 else "test" if index < 8 else "train"
            if folds[group] in (1, 2):
                split = "external_validation" if folds[group] == 1 else "external_test"
            assignments[(group, int(repetition))] = split
    result["split"] = [assignments[(r.group, r.condition_trial)] for r in result.itertuples()]
    result["eligible"] = result.retained & result.condition.isin(conditions) & ~result.cohort_fold.isin([1, 2])
    result["exclusion_reason"] = [
        "artifact"
        if not r.retained
        else "reserved_cohort"
        if r.cohort_fold in (1, 2)
        else "condition_not_selected"
        if r.condition not in conditions
        else ""
        for r in result.itertuples()
    ]
    result["gallery_pool"] = "closed" if profile == "combined-closed" else result.condition
    galleries = []
    for (group, split, pool), rows in result[result.eligible].groupby(["group", "split", "gallery_pool"], sort=True):
        rows = rows.sort_values(["condition_trial", "condition"])
        ids = rows.trial_id.tolist()
        require(len(ids) >= 2, f"Fewer than two candidates: {group}/{split}")
        for member in "ab":
            galleries.append(
                dict(
                    gallery_id=f"{group}:{member}:{pool}:{split}",
                    group=group,
                    member=member,
                    condition=pool,
                    trial_conditions=json.dumps(rows.condition.tolist()),
                    split=split,
                    target="self",
                    trial_ids=json.dumps(ids),
                    candidate_count=len(ids),
                    chance_top1=1 / len(ids),
                    chance_mrr=sum(1 / rank for rank in range(1, len(ids) + 1)) / len(ids),
                    main_summary_eligible=group != 2020,
                )
            )
    return result, pd.DataFrame(galleries)


def split_cache(
    cache: Path, cohort_manifest: Path, output: Path, seed: int = 2042, profile: str = "combined-closed"
) -> dict:
    """Write immutable metadata-only splits using an explicit historical cohort manifest."""
    metadata, trials, windows = validate_cache(cache)
    cohort = json.loads(cohort_manifest.read_text())
    require(cohort.get("dataset") == "circling-local-project1", "Expected a Circling contrastive cohort manifest")
    folds = {int(group): int(fold) for group, fold in cohort["group_folds"].items()}
    splits, galleries = build_protocol(trials, folds, seed, profile)
    windows = windows.drop(columns="split").merge(
        splits[["trial_id", "split", "cohort_fold", "repetition_id", "eligible", "exclusion_reason", "gallery_pool"]],
        on="trial_id",
        validate="many_to_one",
        how="left",
    )
    output.mkdir(parents=True, exist_ok=False)
    frames = {"splits.csv": splits, "windows.csv": windows, "galleries.csv": galleries}
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    result = dict(
        format_version=2,
        operation="split",
        seed=seed,
        profile=profile,
        conditions=["open"] if profile == "open" else ["closed1", "closed2"],
        target="self",
        cache_source=str(cache.resolve()),
        cache_sha256=sha256(cache / "cache.json"),
        cohort_source=str(cohort_manifest.resolve()),
        cohort_sha256=sha256(cohort_manifest),
        group_folds=folds,
        prior_exposure="unknown; inherited reservations are not proof of no prior exposure",
        validation_repetitions_before_artifacts=4,
        test_repetitions_before_artifacts=4,
        trials_per_participant_before_artifacts={
            "train": 12 if profile == "open" else 24,
            "validation": 4 if profile == "open" else 8,
            "test": 4 if profile == "open" else 8,
        },
        assignment="12/4/4 paired repetitions before artifact filtering; all conditions inherit assignment",
        gallery_contract="trial_ids are both query and candidate order; diagonal positives; same member and offset",
        available_crop_offsets_seconds=list(range(0, 15, 2)),
        initial_training_offsets_seconds=[0, 4, 8, 12],
        signal_arrays_verified=False,
        signal_arrays_loaded=False,
        statistics_fitted=False,
        retained_trials=metadata["retained_trials"],
        eligible_trials_by_split=splits[splits.eligible].groupby("split").size().to_dict(),
        candidate_counts=galleries.groupby("split").candidate_count.agg(["min", "max"]).to_dict("index"),
        manifest_sha256={name: sha256(output / name) for name in frames},
        code_sha256=sha256(Path(__file__)),
    )
    (output / "split.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
