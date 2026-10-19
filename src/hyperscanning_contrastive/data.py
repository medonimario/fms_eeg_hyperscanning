"""Audit local Circling recordings and cache symmetrically clean paired windows."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from dotenv import load_dotenv

CHANNELS = (
    "Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 "
    "FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 T7 C5 C3 C1 Cz C2 C4 C6 T8 "
    "TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 P9 P7 P5 P3 P1 Pz P2 P4 P6 P8 P10 "
    "PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz"
).split()
CONDITIONS = ("closed1", "open", "closed2")
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(supplied: Path | None) -> Path:
    """Find the local flat Circling preprocessing directory."""
    load_dotenv(ROOT / ".env", override=False)
    value = supplied or os.environ.get("CIRCLING_DATASET_PATH")
    if not value:
        raise ValueError("Supply --data-dir or set CIRCLING_DATASET_PATH to the local Project 1 dataset.")
    root = Path(value).expanduser().resolve()
    for directory in (root / "preprocessed", root):
        if any(directory.glob("*_ave_epo.fif")):
            return directory
    raise ValueError(f"No local *_ave_epo.fif files found in {root} or its preprocessed directory.")


def discover(directory: Path) -> dict[int, list[Path]]:
    """Require exactly two distinct source recordings per local dyad."""
    groups: dict[int, list[Path]] = {}
    for path in sorted(directory.glob("*_ave_epo.fif")):
        match = re.search(r"pair(\d+)", path.name)
        if match is None:
            raise ValueError(f"Cannot identify local dyad in {path.name}.")
        groups.setdefault(int(match.group(1)), []).append(path)
    if not groups:
        raise ValueError("No Circling recordings found.")
    for group, paths in groups.items():
        if len(paths) != 2 or len({path.resolve() for path in paths}) != 2:
            raise ValueError(f"Dyad {group} requires exactly two distinct participant files; found {paths}.")
    return groups


def read_bad_segments(path: Path) -> dict[str, list[int]]:
    """Read the user's trusted local six-second artifact dictionary and validate its schema."""
    with path.open("rb") as stream:
        raw = pickle.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Artifact file must contain a dictionary.")
    result: dict[str, list[int]] = {}
    for key, values in raw.items():
        if not isinstance(key, str) or not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError(f"Invalid artifact dictionary entry: {key!r}.")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in values):
            raise ValueError(f"Artifact indices must be nonnegative integers: {key}.")
        result[key] = sorted(set(int(v) for v in values))
    return result


def clean_windows(bad_sections: list[int], strategy: str = "max-clean") -> list[tuple[int, int]]:
    """Pack non-overlapping four-second windows into a cropped eighteen-second trial.

    Args:
        bad_sections: Bad section numbers 0, 1, or 2, each representing six seconds.
        strategy: max-clean packs each clean run; fixed-grid retains starts 0, 4, 8, 12.

    Returns:
        Half-open intervals in seconds relative to the two-second crop origin.
    """
    if strategy not in {"max-clean", "fixed-grid"} or any(section not in (0, 1, 2) for section in bad_sections):
        raise ValueError("Invalid strategy or six-second section number.")
    bad = [(section * 6, (section + 1) * 6) for section in sorted(set(bad_sections))]
    if strategy == "fixed-grid":
        return [(s, s + 4) for s in range(0, 15, 4) if not any(s < b and s + 4 > a for a, b in bad)]
    runs, cursor = [], 0
    for begin, end in bad:
        if cursor < begin:
            runs.append((cursor, begin))
        cursor = end
    if cursor < 18:
        runs.append((cursor, 18))
    return [(s, s + 4) for begin, end in runs for s in range(begin, end - 3, 4)]


def validate_pair(first: mne.BaseEpochs, second: mne.BaseEpochs, group: int) -> None:
    """Fail on ambiguous trial correspondence rather than pairing surviving row positions.

    Args:
        first: First participant's original epochs, before practice removal.
        second: Second participant's original epochs on the same acquisition clock.
        group: Local dyad identifier for diagnostics.

    Raises:
        ValueError: If timing, event identities, selection, or required channels differ.
    """
    if not np.array_equal(first.events, second.events) or not np.array_equal(first.selection, second.selection):
        raise ValueError(f"Dyad {group}: events/selection differ; original trial mapping must be resolved explicitly.")
    if first.event_id != second.event_id:
        raise ValueError(f"Dyad {group}: event dictionaries differ.")
    if not np.array_equal(first.times, second.times) or first.info["sfreq"] != second.info["sfreq"]:
        raise ValueError(f"Dyad {group}: sampling clocks or epoch time axes differ.")
    if not np.array_equal(first.selection, np.arange(len(first))):
        raise ValueError(f"Dyad {group}: previously dropped/reindexed epochs make six-second indices ambiguous.")
    if len(first.events) > 1 and np.any(np.diff(first.events[:, 0]) <= 0):
        raise ValueError(f"Dyad {group}: event samples must be strictly increasing.")
    for epochs in (first, second):
        if len(set(epochs.ch_names)) != len(epochs.ch_names) or not set(CHANNELS) <= set(epochs.ch_names):
            raise ValueError(f"Dyad {group}: missing or duplicate EEG channels.")
        if any(kind != "eeg" for kind in epochs.get_channel_types(picks=CHANNELS)):
            raise ValueError(f"Dyad {group}: all required channels must be typed as EEG for microvolt conversion.")
        sfreq = epochs.info["sfreq"]
        if epochs.tmin > 2 or epochs.tmax < 20 - 1 / sfreq:
            raise ValueError(f"Dyad {group}: epochs must contain the entire [2, 20) second analysis interval.")
        if not np.isclose(sfreq, round(sfreq)) or not np.isclose(
            (2 - epochs.tmin) * sfreq, round((2 - epochs.tmin) * sfreq)
        ):
            raise ValueError(f"Dyad {group}: interval boundaries must align with original samples.")


def plan_pair(
    first: mne.BaseEpochs,
    second: mne.BaseEpochs,
    group: int,
    bad: dict[str, list[int]],
    conditions: tuple[str, ...] = CONDITIONS,
    strategy: str = "max-clean",
) -> tuple[list[dict], list[dict]]:
    """Create a pair-symmetric window manifest and a complete per-trial exclusion census."""
    validate_pair(first, second, group)
    practice = 12 if group == 2030 else 6
    if len(first) <= practice:
        raise ValueError(f"Dyad {group}: no experimental trials after {practice} practice epochs.")
    rows, trials = [], []
    selected_codes = [first.event_id[name] for name in conditions if name in first.event_id]
    if len(set(selected_codes)) != len(selected_codes):
        raise ValueError(f"Dyad {group}: requested conditions share event codes.")
    for condition in conditions:
        if condition not in first.event_id:
            raise ValueError(f"Dyad {group}: missing requested condition {condition}.")
        code = first.event_id[condition]
        indices = np.flatnonzero(first.events[:, 2] == code)
        expected_practice = practice // 3
        if np.count_nonzero(indices < practice) != expected_practice:
            raise ValueError(f"Dyad {group}: unexpected practice condition counts for {condition}.")
        indices = indices[indices >= practice]
        if not len(indices):
            raise ValueError(f"Dyad {group}: no experimental epochs for {condition}.")
        key = f"{group}_{condition}"
        marked = bad.get(key, [])
        if any(index >= len(indices) * 3 for index in marked):
            raise ValueError(f"{key}: artifact index exceeds the available condition trials.")
        for trial, source_row in enumerate(indices):
            sections = [index % 3 for index in marked if index // 3 == trial]
            intervals = clean_windows(sections, strategy)
            trial_id = f"{group}:{condition}:{int(first.selection[source_row])}"
            common = dict(
                group=group,
                condition=condition,
                condition_trial=trial,
                trial_id=trial_id,
                source_epoch=int(first.selection[source_row]),
                source_row=int(source_row),
                event_sample=int(first.events[source_row, 0]),
                event_code=int(code),
                source_sfreq=float(first.info["sfreq"]),
                source_tmin=float(first.tmin),
                practice_epochs_removed=practice,
                bad_key_present=key in bad,
                bad_sections=";".join(map(str, sections)),
            )
            trials.append(dict(common, retained_windows=len(intervals), clean_seconds=18 - 6 * len(sections)))
            for start, end in intervals:
                rows.append(
                    dict(
                        common,
                        pair_id=f"{trial_id}:{start}",
                        crop_start_s=start,
                        crop_end_s=end,
                        onset_s=start + 2,
                        end_s=end + 2,
                        start_sample=int(round((start + 2 - first.tmin) * first.info["sfreq"])),
                        stop_sample=int(round((end + 2 - first.tmin) * first.info["sfreq"])),
                    )
                )
    return rows, trials


def extract_pair(first: mne.BaseEpochs, second: mne.BaseEpochs, rows: list[dict]) -> np.ndarray:
    """Extract aligned windows in float32 microvolts without applying model input scaling."""
    result = np.empty((len(rows), 2, 64, 800), dtype=np.float32)
    for member, epochs in enumerate((first, second)):
        picks = [epochs.ch_names.index(name) for name in CHANNELS]
        trial_data: np.ndarray | None = None
        last_row = None
        for index, row in enumerate(rows):
            if row["source_row"] != last_row:
                trial_data = epochs[row["source_row"] : row["source_row"] + 1].get_data(picks=picks, units="uV")[0]
                last_row = row["source_row"]
            window = trial_data[:, row["start_sample"] : row["stop_sample"]]
            if window.shape != (64, int(round(4 * row["source_sfreq"]))):
                raise ValueError(f"Incomplete source window: {row['pair_id']}.")
            if row["source_sfreq"] != 200:
                window = mne.filter.resample(window, up=200, down=row["source_sfreq"], npad="auto", verbose=False)
            if window.shape != (64, 800) or not np.isfinite(window).all():
                raise ValueError(f"Invalid signal in {row['pair_id']}.")
            result[index, member] = window
    return result


def prepare(
    directory: Path,
    destination: Path,
    bad_path: Path,
    write_signals: bool = False,
    conditions: tuple[str, ...] = CONDITIONS,
    strategy: str = "max-clean",
) -> dict:
    """Audit all dyads, optionally writing a new immutable cache after successful validation.

    Args:
        directory: Local flat preprocessing directory.
        destination: New output directory; never overwrite existing outputs.
        bad_path: Trusted local artifact dictionary with six-second indices.
        write_signals: Whether to extract and save paired signal arrays as well as metadata.
        conditions: Explicit conditions to retain.
        strategy: Non-overlapping window placement policy.

    Returns:
        Completion metadata including usable-pair counts and source provenance.
    """
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= set(CONDITIONS):
        raise ValueError(f"Select unique conditions from {CONDITIONS}.")
    if strategy not in {"max-clean", "fixed-grid"}:
        raise ValueError(f"Unknown window strategy: {strategy}.")
    groups = discover(directory)
    bad = read_bad_segments(bad_path)
    destination.mkdir(parents=True, exist_ok=False)
    shuffled = np.array(sorted(groups))
    np.random.RandomState(42).shuffle(shuffled)
    fold_count = min(10, len(groups))
    folds = {int(group): fold for fold, chunk in enumerate(np.array_split(shuffled, fold_count)) for group in chunk}
    all_rows, all_trials, files = [], [], []
    used_keys = set()
    for group, paths in groups.items():
        print(f"Auditing dyad {group}", flush=True)
        first, second = [mne.read_epochs(path, preload=False, verbose=False) for path in paths]
        rows, trials = plan_pair(first, second, group, bad, conditions, strategy)
        used_keys.update(f"{group}_{condition}" for condition in conditions)
        for member, (path, epochs) in enumerate(zip(paths, (first, second), strict=True)):
            files.append(
                dict(
                    group=group,
                    member=member,
                    participant=path.name.removesuffix("_ave_epo.fif"),
                    source=str(path.resolve()),
                    source_bytes=path.stat().st_size,
                    source_mtime_ns=path.stat().st_mtime_ns,
                    source_sha256=sha256(path) if write_signals else None,
                    epochs=len(epochs),
                    sfreq=epochs.info["sfreq"],
                    fold=folds[group],
                )
            )
        filename = f"pair_{group}.npy"
        if write_signals:
            signal = extract_pair(first, second, rows)
            np.save(destination / filename, signal, allow_pickle=False)
        for index, row in enumerate(rows):
            row.update(
                fold=folds[group],
                participant_a=paths[0].name.removesuffix("_ave_epo.fif"),
                participant_b=paths[1].name.removesuffix("_ave_epo.fif"),
                array=filename if write_signals else "",
                array_index=index,
            )
        all_rows.extend(rows)
        all_trials.extend(dict(row, fold=folds[group]) for row in trials)
    if not all_rows:
        raise ValueError("No paired windows remain; output is incomplete.")
    windows = pd.DataFrame(all_rows)
    windows.to_csv(destination / "windows.csv", index=False)
    pd.DataFrame(all_trials).to_csv(destination / "trials.csv", index=False)
    pd.DataFrame(files).to_csv(destination / "files.csv", index=False)
    census = windows.groupby(["group", "fold", "condition"]).size().rename("paired_windows")
    census = (
        pd.DataFrame(all_trials)
        .groupby(["group", "fold", "condition"])
        .agg(trials=("trial_id", "size"), usable_trials=("retained_windows", lambda counts: int((counts > 0).sum())))
        .join(census)
        .fillna({"paired_windows": 0})
        .reset_index()
    )
    census.to_csv(destination / "census.csv", index=False)
    summary = dict(
        format_version=1,
        dataset="circling-local-project1",
        operation="prepare" if write_signals else "audit",
        pairing="strict identical original events, selection and time axes; no dropped source epochs",
        conditions=list(conditions),
        channels=CHANNELS,
        units="microvolts",
        model_input_divisor=100,
        sfreq=200,
        window_seconds=4,
        window_strategy=strategy,
        crop_interval_seconds=[2, 20],
        artifact_segment_seconds=6,
        artifact_segments_per_trial=3,
        artifact_index_origin="condition-local experimental trial after practice removal and two-second crop",
        artifact_policy="exclude positive-duration overlap for both participants; touching endpoints are allowed",
        missing_artifact_keys=sorted(used_keys - set(bad)),
        unused_artifact_keys=sorted(set(bad) - used_keys),
        bad_segments_source=str(bad_path.resolve()),
        bad_segments_sha256=sha256(bad_path),
        group_count=len(groups),
        paired_window_count=len(windows),
        group_folds=folds,
        fold_seed=42,
        fold_count=fold_count,
        evaluation_ready=fold_count == 10 and all(census.paired_windows > 0),
        mne_version=mne.__version__,
        signal_validation="finite float32 (pairs, 2, 64, 800)"
        if write_signals
        else "not performed; metadata audit only",
        manifest_sha256={
            name: sha256(destination / name) for name in ("windows.csv", "trials.csv", "files.csv", "census.csv")
        },
        array_sha256={path.name: sha256(path) for path in sorted(destination.glob("*.npy"))},
    )
    marker = "cache.json" if write_signals else "audit.json"
    (destination / marker).write_text(json.dumps(summary, indent=2) + "\n")
    return summary
