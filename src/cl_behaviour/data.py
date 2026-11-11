"""Prepare whole-trial-clean Circling EEG and original movement phases."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from hyperscanning_contrastive.data import (
    CHANNELS,
    CONDITIONS,
    discover,
    extract_pair,
    plan_pair,
    read_bad_segments,
    sha256,
)
from mTRF.common import PAIRS


def eeg_path(directory: Path, pair: int, member: str) -> Path:
    """Resolve explicit a/b filenames in the flat preprocessing directory."""
    paths = [directory / f"pair{pair}{sep}{member}_ave_epo.fif" for sep in ("", "_", "-")]
    paths = [path for path in paths if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one recording for {pair}{member}: {paths}")
    return paths[0]


def movement_phase(phase: tuple, pair: int) -> np.ndarray:
    """Apply mTRF's phase shape, extra-trial and crop contract without importing its solver."""
    values = np.asarray(phase)
    expected = (2, 3, 21 if pair == 2450 else 20, 4800)
    if values.shape != expected:
        raise ValueError(f"Pair {pair}: phase shape {values.shape}, expected {expected}")
    if pair == 2450:
        values = np.delete(values, 10, axis=2)
    return values[..., 480:]


def retained_trials(bad: list[int]) -> list[int]:
    """Reproduce the reference notebook mapping and its zero-based membership test exactly."""
    excluded = {(int(index) - 1) % 20 + 1 for index in bad}
    return [i for i in range(20) if i not in excluded]


def aligned_plan(first: mne.BaseEpochs, second: mne.BaseEpochs, pair: int, bad: dict) -> tuple[list, list]:
    """Reject complete paired trials and link eight overlapping windows to each survivor."""
    for condition in CONDITIONS:
        if f"{pair}_{condition}" not in bad:
            raise ValueError(f"Missing artifact key: {pair}_{condition}")
    # Reuse EEG identity checks only; the contrastive artifact mapping is not applicable here.
    _, trials = plan_pair(first, second, pair, {f"{pair}_{c}": [] for c in CONDITIONS})
    windows = []
    for condition in CONDITIONS:
        selected = [row for row in trials if row["condition"] == condition]
        if len(selected) != 20:
            raise ValueError(f"Pair {pair}: expected 20 aligned {condition} trials.")
        marked = bad[f"{pair}_{condition}"]
        if any(index > 60 for index in marked):
            raise ValueError(f"Pair {pair}: artifact index exceeds 60 condition segments.")
        historical = retained_trials(marked)
        for row in selected:
            trial = row["condition_trial"]
            keep = trial in historical
            row.pop("bad_sections")
            row.pop("clean_seconds")
            row.update(
                retained_windows=8 if keep else 0,
                retained=keep,
                reason="" if keep else "reference_notebook_bad_trial",
                artifact_indices=";".join(str(i) for i in marked if (i - 1) % 20 + 1 == trial),
                historical_mtrf_retained=trial in historical,
                movement_original_trial=trial + int(pair == 2450 and trial >= 10),
                main_summary_eligible=pair != 2020,
                split="unassigned",
            )
            if not keep:
                continue
            for start in range(0, 15, 2):
                windows.append(
                    dict(
                        row,
                        pair_id=f"{row['trial_id']}:{start}",
                        crop_start_s=start,
                        crop_end_s=start + 4,
                        onset_s=start + 2,
                        end_s=start + 6,
                        start_sample=round((start + 2 - first.tmin) * first.info["sfreq"]),
                        stop_sample=round((start + 6 - first.tmin) * first.info["sfreq"]),
                        movement_start_sample=start * 240,
                        movement_stop_sample=(start + 4) * 240,
                    )
                )
    return windows, trials


def prepare(directory: Path, phase_path: Path, bad_path: Path, output: Path, write: bool = False) -> dict:
    """Validate source identity and write a fresh audit or paired cache.

    Args:
        directory: Flat preprocessed EEG directory.
        phase_path: Trusted original movement phase-list pickle in notebook cohort order.
        bad_path: Trusted six-second artifact dictionary.
        output: New destination directory.
        write: Whether to extract EEG and save retained movement arrays.

    Returns:
        Completion metadata, published only after all pairs pass validation.
    """
    groups = discover(directory)
    if set(groups) != set(PAIRS):
        raise ValueError("Expected the complete original 52-dyad cohort.")
    with phase_path.open("rb") as stream:
        phases = pickle.load(stream)
    if not isinstance(phases, (list, tuple)) or len(phases) != len(PAIRS):
        raise ValueError("Expected original phase_list with 52 ordered dyads.")
    bad = read_bad_segments(bad_path)
    output.mkdir(parents=True, exist_ok=False)
    all_windows, all_trials, files = [], [], []
    for pair, raw_phase in zip(PAIRS, phases, strict=True):
        print(f"{'Preparing' if write else 'Auditing'} pair {pair}", flush=True)
        phase = movement_phase(raw_phase, pair)
        if not np.isfinite(phase).all():
            raise ValueError(f"Nonfinite phase for pair {pair}")
        paths = [eeg_path(directory, pair, member) for member in "ab"]
        epochs = [mne.read_epochs(path, preload=False, verbose=False) for path in paths]
        windows, trials = aligned_plan(*epochs, pair, bad)
        eeg_name, movement_name = f"{pair}_eeg.npy", f"{pair}_phase.npy"
        retained = [row for row in trials if row["retained"]]
        indices = {row["trial_id"]: i for i, row in enumerate(retained)}
        for row in trials:
            row.update(
                movement_array=movement_name if write else "", movement_array_index=indices.get(row["trial_id"], -1)
            )
        for i, row in enumerate(windows):
            row.update(
                eeg_array=eeg_name if write else "",
                eeg_array_index=i,
                movement_array=movement_name if write else "",
                movement_array_index=indices[row["trial_id"]],
            )
        if write:
            np.save(output / eeg_name, extract_pair(*epochs, windows), allow_pickle=False)
            values = (
                np.stack([phase[:, CONDITIONS.index(row["condition"]), row["condition_trial"]] for row in retained])
                if retained
                else np.empty((0, 2, 4320))
            )
            np.save(output / movement_name, values, allow_pickle=False)
        for member, path in zip("ab", paths, strict=True):
            files.append(dict(pair=pair, member=member, source=str(path.resolve()), source_sha256=sha256(path)))
        all_windows.extend(windows)
        all_trials.extend(trials)
    if not all_windows:
        raise ValueError("No retained trials; output is incomplete.")
    frames = {
        "windows.csv": pd.DataFrame(all_windows),
        "trials.csv": pd.DataFrame(all_trials),
        "files.csv": pd.DataFrame(files),
    }
    frames["census.csv"] = (
        frames["trials.csv"]
        .groupby(["group", "condition"])
        .agg(
            total_trials=("trial_id", "size"),
            retained_trials=("retained", "sum"),
            paired_windows=("retained_windows", "sum"),
        )
        .reset_index()
    )
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    result = dict(
        format_version=2,
        dataset="circling-eeg-behaviour",
        operation="prepare" if write else "audit",
        channels=CHANNELS,
        members=["a", "b"],
        conditions=list(CONDITIONS),
        eeg_units="microvolts",
        eeg_sfreq=200,
        model_input_divisor=100,
        eeg_array_axes=["window", "member", "channel", "sample"],
        movement_array_axes=["trial", "member", "sample"],
        movement_sfreq=240,
        movement_representation="original vertical Hilbert phase in radians; no resampling",
        trial_interval_seconds=[2, 20],
        window_seconds=4,
        stride_seconds=2,
        artifact_policy="mtrf-notebook-exact",
        artifact_trial_mapping="((index - 1) % 20) + 1 compared directly with range(20)",
        artifact_reference="src/mTRF/example_EEG_mTRF.ipynb: segment_to_epoch and good_e",
        artifact_boundary_policy="preserve notebook: mapped value 20 excludes no zero-based trial",
        artifact_unmatched_indices={
            key: [i for i in values if (i - 1) % 20 + 1 == 20]
            for key, values in bad.items()
            if any((i - 1) % 20 + 1 == 20 for i in values)
        },
        split="unassigned; no statistics fitted",
        pair2020_policy="retained; main_summary_eligible=false",
        phase_source=str(phase_path.resolve()),
        phase_sha256=sha256(phase_path),
        bad_segments_source=str(bad_path.resolve()),
        bad_segments_sha256=sha256(bad_path),
        retained_trials=sum(row["retained"] for row in all_trials),
        paired_windows=len(all_windows),
        historical_exclusion_disagreements=sum(
            row["retained"] != row["historical_mtrf_retained"] for row in all_trials
        ),
        mne_version=mne.__version__,
        numpy_version=np.__version__,
        code_sha256={name: sha256(Path(name)) for name in [__file__]},
        manifest_sha256={name: sha256(output / name) for name in frames},
        array_sha256={p.name: sha256(p) for p in sorted(output.glob("*.npy"))},
    )
    (output / ("cache.json" if write else "audit.json")).write_text(json.dumps(result, indent=2) + "\n")
    return result
