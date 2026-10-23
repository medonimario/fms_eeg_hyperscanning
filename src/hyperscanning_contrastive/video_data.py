"""Audit and prepare independent, stimulus-timed video EEG recordings."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import resample_poly

from .data import CHANNELS, sha256

VIDEOS = (0, 3, 4, 5, 6, 7, 8, 9, 10)
EXCLUDED = {"solo11_A_1": "confirmed_faulty"}
PROVISIONAL = {"pair4_together_A_4_p2", "pair9_together_A_9_p2"}


def subject_metadata(name: str, condition: str) -> dict:
    """Parse identity without conflating counterbalance, configuration, and participant."""
    pattern = r"solo(\d+)_([AB])_(\d+)" if condition == "solo" else r"pair(\d+)_together_([AB])_(\d+)_p([12])"
    match = re.fullmatch(pattern, name)
    if match is None:
        raise ValueError(f"Unrecognized {condition} participant: {name}")
    number, profile, config, *member = match.groups()
    group = f"solo{number}" if condition == "solo" else f"pair{number}"
    return {
        "participant_id": name,
        "condition": condition,
        "independence_group": group,
        "pair_id": group if member else "",
        "member": int(member[0]) if member else 0,
        "counterbalance": profile,
        "config_index": int(config),
        "babble_id": "unknown",
    }


def noise_level(video: int, profile: str) -> str:
    """Map counterbalance and movie to experimental noise level."""
    if video not in VIDEOS or profile not in ("A", "B"):
        raise ValueError("Unknown video or counterbalance.")
    if video == 0:
        return "none"
    return "low" if (video <= 6) == (profile == "A") else "high"


def load_recording(path: Path) -> tuple[np.ndarray, dict]:
    """Validate a trusted NPZ and return channel-ordered EEG in microvolts."""
    payload = path.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=True) as archive:
        required = {
            "data",
            "ch_names",
            "sfreq",
            "video_number",
            "video_start_code",
            "video_end_code",
            "start_sample",
            "end_sample",
        }
        if not required <= set(archive.files):
            raise ValueError(f"Missing NPZ fields: {path}")
        names = [str(x) for x in archive["ch_names"]]
        data = archive["data"]
        if data.ndim != 2 or data.shape[1] != 64 or data.dtype.kind != "f" or len(data) == 0:
            raise ValueError(f"Invalid EEG shape or dtype: {path}")
        if len(names) != 64 or len(set(names)) != 64 or set(names) != set(CHANNELS):
            raise ValueError(f"Invalid channel identities: {path}")
        metadata = {}
        for field in required - {"data", "ch_names"}:
            value = archive[field]
            if value.size != 1 or not np.isfinite(float(value.item())):
                raise ValueError(f"Invalid scalar {field}: {path}")
            metadata[field] = float(value.item())
        if metadata["sfreq"] != 256:
            raise ValueError(f"Expected 256Hz: {path}")
        for field in required - {"data", "ch_names", "sfreq"}:
            if not metadata[field].is_integer():
                raise ValueError(f"Noninteger {field}: {path}")
            metadata[field] = int(metadata[field])
        video = metadata["video_number"]
        if video not in VIDEOS or path.stem != f"video{video}":
            raise ValueError(f"Video identity mismatch: {path}")
        if metadata["start_sample"] < 0 or metadata["end_sample"] <= metadata["start_sample"]:
            raise ValueError(f"Invalid original trigger indices: {path}")
        discrepancy = len(data) / 256 - (metadata["end_sample"] - metadata["start_sample"]) / 2048
        if abs(discrepancy) > 1 / 256:
            raise ValueError(f"Trigger duration disagrees with signal length: {path}")
        ordered = np.ascontiguousarray(data[:, [names.index(c) for c in CHANNELS]].T * 1e6)
    metadata.update(
        {
            "source_path": str(path.resolve()),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "signal_sha256": hashlib.sha256(ordered.tobytes()).hexdigest(),
            "native_samples": ordered.shape[1],
            "duration_seconds": ordered.shape[1] / 256,
            "duration_discrepancy_seconds": discrepancy,
            "native_channel_order": json.dumps(names),
        }
    )
    return ordered, metadata


def window_quality(data: np.ndarray, amplitude_uv: float, flat_std_uv: float) -> list[str]:
    """Flag gross residual artifacts without learning thresholds from evaluation data."""
    reasons = []
    if not np.isfinite(data).all():
        return ["nonfinite"]
    if np.abs(data).max() > amplitude_uv:
        reasons.append("amplitude")
    if np.std(data, axis=1).min() < flat_std_uv:
        reasons.append("flat_channel")
    if np.abs(np.diff(data, axis=1)).max() > amplitude_uv:
        reasons.append("discontinuity")
    return reasons


def prepare_videos(
    data_dir: Path,
    output: Path,
    prepare: bool = False,
    edge_guard: float = 4.0,
    amplitude_uv: float = 200.0,
    flat_std_uv: float = 0.01,
    include_provisional: bool = False,
) -> dict:
    """Scan all supplied signals, retain absolute stimulus positions, and optionally cache continuous EEG."""
    if not np.isfinite([edge_guard, amplitude_uv, flat_std_uv]).all():
        raise ValueError("QC settings must be finite.")
    if edge_guard < 0 or amplitude_uv <= 0 or flat_std_uv < 0:
        raise ValueError("Invalid edge guard or QC thresholds.")
    data_dir = data_dir.expanduser().resolve()
    subjects = []
    for condition in ("solo", "together"):
        directory = data_dir / f"{condition}_condition"
        if not directory.is_dir():
            raise ValueError(f"Missing source directory: {directory}")
        for folder in sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")):
            row = subject_metadata(folder.name, condition)
            row["exclusion_reason"] = EXCLUDED.get(folder.name, "")
            if folder.name in PROVISIONAL and not include_provisional:
                row["exclusion_reason"] = "provisional_noisy"
            row["folder"] = str(folder)
            subjects.append(row)
    if not subjects:
        raise ValueError("No video participants found.")
    subject_frame = pd.DataFrame(subjects)
    identity = ["independence_group", "member"]
    if subject_frame.duplicated(identity).any():
        raise ValueError("Repeated participant/group identity; resolve sessions before splitting.")
    for _, group in subject_frame[subject_frame.condition == "together"].groupby("pair_id"):
        if group.counterbalance.nunique() != 1 or group.config_index.nunique() != 1:
            raise ValueError("Together partner configuration mismatch.")
    output.mkdir(parents=True, exist_ok=False)
    if prepare:
        (output / "signals").mkdir()
    files, missing, windows = [], [], []
    signal_owners = {}
    for subject in subjects:
        folder = Path(subject["folder"])
        paths = sorted(folder.glob("video*.npz"))
        present = set()
        for path in paths:
            data, metadata = load_recording(path)
            video = metadata["video_number"]
            present.add(video)
            finite = bool(np.isfinite(data).all())
            reason = subject["exclusion_reason"] or ("nonfinite_recording" if not finite else "")
            digest = metadata["signal_sha256"]
            duplicate = signal_owners.get(digest, "")
            signal_owners[digest] = str(path)
            if duplicate:
                reason = reason or "duplicate_signal"
            array_name = f"signals/{subject['participant_id']}_video{video}.npy"
            record = {
                **subject,
                **metadata,
                "noise_level": noise_level(video, subject["counterbalance"]),
                "finite": finite,
                "exclusion_reason": reason,
                "duplicate_of": duplicate,
                "array_path": array_name if prepare and not reason else "",
                "array_sha256": "",
                "resampled_samples": 0,
            }
            if finite:
                record.update(
                    {
                        "std_uv": float(data.std()),
                        "max_abs_uv": float(abs(data).max()),
                        "max_car_residual_uv": float(abs(data.mean(0)).max()),
                    }
                )
            if prepare and not reason:
                resampled = resample_poly(data, 25, 32, axis=1).astype(np.float32)
                if not np.isfinite(resampled).all():
                    raise ValueError(f"Nonfinite resampling output: {path}")
                np.save(output / array_name, resampled, allow_pickle=False)
                record["array_sha256"] = sha256(output / array_name)
                record["resampled_samples"] = resampled.shape[1]
                del resampled
            files.append(record)
            for start in range(0, data.shape[1] - 1024 + 1, 1024):
                quality = window_quality(data[:, start : start + 1024], amplitude_uv, flat_std_uv)
                windows.append(
                    {
                        **{
                            k: subject[k]
                            for k in (
                                "participant_id",
                                "condition",
                                "independence_group",
                                "pair_id",
                                "member",
                                "counterbalance",
                                "config_index",
                                "babble_id",
                            )
                        },
                        "video_id": video,
                        "noise_level": record["noise_level"],
                        "stimulus_window_id": f"video{video}_sample{start}",
                        "start_seconds": start / 256,
                        "end_seconds": (start + 1024) / 256,
                        "native_start": start,
                        "native_stop": start + 1024,
                        "cache_start": start * 25 // 32,
                        "cache_stop": start * 25 // 32 + 800,
                        "recording_exclusion": reason,
                        "qc_flags": ";".join(quality),
                        "array_path": record["array_path"],
                        "source_sha256": record["source_sha256"],
                        "split": "unassigned",
                    }
                )
            print(f"Scanned {subject['participant_id']} video{video}: {data.shape[1]} samples", flush=True)
        for video in sorted(set(VIDEOS) - present):
            missing.append({"participant_id": subject["participant_id"], "video_id": video})
    if not files or not windows:
        raise ValueError("No complete four-second video windows found.")
    frame, rows = pd.DataFrame(files), pd.DataFrame(windows)
    common = frame[frame.exclusion_reason == ""].groupby("video_number").duration_seconds.min().to_dict()
    rows["within_common_support"] = [
        start >= edge_guard and stop <= common.get(video, 0) - edge_guard
        for video, start, stop in zip(rows.video_id, rows.start_seconds, rows.end_seconds, strict=True)
    ]
    rows["eligible"] = (rows.recording_exclusion == "") & (rows.qc_flags == "") & rows.within_common_support
    rows["reason"] = [
        exclusion or quality or ("edge_or_common_tail" if not support else "usable")
        for exclusion, quality, support in zip(
            rows.recording_exclusion, rows.qc_flags, rows.within_common_support, strict=True
        )
    ]
    pair_rows = []
    for (pair, video), group in frame[frame.condition == "together"].groupby(["pair_id", "video_number"]):
        members = group.set_index("member")
        both = set(members.index) == {1, 2}
        pair_rows.append(
            {
                "pair_id": pair,
                "video_id": video,
                "both_present": both,
                "both_included": both and bool((members.exclusion_reason == "").all()),
                "equal_length": both and members.native_samples.nunique() == 1,
                "p2_minus_p1_start_samples": int(members.loc[2, "start_sample"] - members.loc[1, "start_sample"])
                if both
                else None,
            }
        )
    keys = ["condition", "counterbalance", "noise_level", "video_id"]
    census = (
        rows.groupby(keys)
        .agg(
            person_windows=("eligible", "size"),
            usable_windows=("eligible", "sum"),
            participants=("participant_id", "nunique"),
        )
        .reset_index()
    )
    coverage = (
        rows[rows.eligible]
        .groupby(keys + ["stimulus_window_id"])
        .agg(available_people=("participant_id", "nunique"), independent_groups=("independence_group", "nunique"))
        .reset_index()
    )
    tables = {
        "participants.csv": subject_frame,
        "files.csv": frame,
        "windows.csv": rows,
        "missing.csv": pd.DataFrame(missing, columns=["participant_id", "video_id"]),
        "pairs.csv": pd.DataFrame(pair_rows),
        "census.csv": census,
        "moment_coverage.csv": coverage,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    report = {
        "format_version": 1,
        "dataset": "shared-stimulus-video",
        "operation": "prepare" if prepare else "audit",
        "source": str(data_dir),
        "participants": len(subjects),
        "recordings": len(files),
        "missing_recordings": len(missing),
        "included_participants": int(frame.loc[frame.exclusion_reason == "", "participant_id"].nunique()),
        "person_windows": len(rows),
        "usable_person_windows": int(rows.eligible.sum()),
        "unique_usable_stimulus_moments": int(rows.loc[rows.eligible, "stimulus_window_id"].nunique()),
        "window_reasons": rows.reason.value_counts().to_dict(),
        "channels": CHANNELS,
        "units": "microvolts",
        "model_input_divisor": 100,
        "source_sfreq": 256,
        "sfreq": 200,
        "window_seconds": 4,
        "edge_guard_seconds": edge_guard,
        "amplitude_threshold_uv": amplitude_uv,
        "flat_channel_std_threshold_uv": flat_std_uv,
        "qc_policy": "native-window max amplitude, any-channel flatness, sample jumps; nonfinite rejects recording",
        "include_provisional": include_provisional,
        "confirmed_exclusions": EXCLUDED,
        "provisional_exclusions": sorted(PROVISIONAL),
        "common_duration_seconds": common,
        "timing": "reported photodiode alignment; segment-relative time; no shifts or time warping",
        "babble_identity": "unknown; A/B is noise-level counterbalance, not babble waveform",
        "resampling": "scipy.signal.resample_poly, up25/down32, default Kaiser anti-alias filter, continuous segment",
        "split_policy": "unassigned; Stage0 does not construct training or evaluation splits",
        "evaluation_ready": False,
        "signal_validation": "all files scanned, including excluded recordings",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "manifest_sha256": {name: sha256(output / name) for name in tables},
        "arrays_written": int((frame.array_path != "").sum()),
    }
    (output / ("cache.json" if prepare else "audit.json")).write_text(json.dumps(report, indent=2) + "\n")
    return report
