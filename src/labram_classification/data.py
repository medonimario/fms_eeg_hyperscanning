"""Dataset-specific preprocessing, reusable disk cache, and leakage-free folds."""

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
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

CHANNELS = (
    "Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 "
    "FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 T7 C5 C3 C1 Cz C2 C4 C6 T8 "
    "TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 P9 P7 P5 P3 P1 Pz P2 P4 P6 P8 P10 "
    "PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz"
).split()
CONTRASTS = {
    "solo-coordination": {"Solo": 0, "Coordination": 1},
    "solo-spontaneous": {"Solo": 0, "Spontaneous": 1},
    "spontaneous-coordination": {"Spontaneous": 0, "Coordination": 1},
}
ENVIRONMENTS = {"mg": "MG_DATASET_PATH", "fg": "FG_DATASET_PATH", "circling": "CIRCLING_DATASET_PATH"}
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """Hash a file in bounded chunks for reproducible provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_directory(dataset: str, supplied: Path | None) -> Path:
    """Resolve an explicit path or dataset environment variable without exposing other variables."""
    load_dotenv(ROOT / ".env", override=False)
    value = supplied or os.environ.get(ENVIRONMENTS[dataset])
    if not value:
        raise ValueError(f"Set {ENVIRONMENTS[dataset]} or pass --data-dir.")
    root = Path(value).expanduser().resolve()
    pattern = "*.set" if dataset == "mg" else "*-epo.fif" if dataset == "fg" else "*_ave_epo.fif"
    for candidate in (root, root / "preprocessed"):
        if any(candidate.glob(pattern)):
            return candidate
    raise FileNotFoundError(f"No {pattern} files found under {root} or its preprocessed directory.")


def discover(directory: Path, dataset: str, contrast: str) -> pd.DataFrame:
    """Identify files and reproduce sorted, seed-42 shuffled ten-fold group assignment."""
    pattern = "*.set" if dataset == "mg" else "*-epo.fif" if dataset == "fg" else "*_ave_epo.fif"
    rows = []
    for path in sorted(directory.glob(pattern)):
        condition = "mixed"
        if dataset == "mg":
            match = re.fullmatch(r"s_([12])(\d{2})_(Solo|Spontaneous|Coordination)\.set", path.name)
            if match is None:
                raise ValueError(f"Unexpected MirrorGame filename: {path.name}")
            member, group, condition = match.groups()
            if condition not in CONTRASTS[contrast]:
                continue
            participant = member + group
        elif dataset == "fg":
            match = re.match(r"(\d+)([A-Za-z])", path.name)
            if match is None:
                raise ValueError(f"Cannot identify FG triad and participant: {path.name}")
            group, member = match.groups()
            participant = group + member.upper()
        else:
            match = re.search(r"pair(\d+)", path.name)
            if match is None:
                raise ValueError(f"Cannot identify Circling dyad: {path.name}")
            group = match.group(1)
            participant = path.name.split("_ave_epo.fif")[0]
        rows.append(dict(source=str(path), group=int(group), participant=participant, condition=condition))
    if not rows:
        raise ValueError("No matching files found.")
    frame = pd.DataFrame(rows)
    groups = np.array(sorted(frame.group.unique()))
    if len(groups) < 10:
        raise ValueError("Ten-fold evaluation requires at least ten distinct groups.")
    np.random.RandomState(42).shuffle(groups)
    assignments = {}
    for fold, (_, indices) in enumerate(KFold(10).split(groups)):
        assignments.update({int(group): fold for group in groups[indices]})
    frame["fold"] = frame.group.map(assignments)
    if dataset == "mg":
        for group, files in frame.groupby("group"):
            if len(files) != 4 or files.participant.nunique() != 2 or files.condition.nunique() != 2:
                raise ValueError(f"MirrorGame dyad {group} lacks complete two-participant/two-condition coverage.")
    return frame


def ordered(epochs: mne.BaseEpochs) -> mne.BaseEpochs:
    """Select the original 64-channel order, failing rather than silently changing inputs."""
    missing = set(CHANNELS) - set(epochs.ch_names)
    if missing:
        raise ValueError(f"Missing required channels: {sorted(missing)}")
    return epochs.reorder_channels(CHANNELS)


def extract_windows(
    epochs: mne.BaseEpochs,
    dataset: str,
    group: int,
    condition: str,
    contrast: str,
    bad_segments: dict,
    fg_endpoint: str = "strict",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Reproduce the original dataset-specific operations and return microvolt windows.

    Args:
        epochs: Loaded preprocessed epochs; a copy is modified.
        dataset: mg, fg, or circling.
        group: Dyad/triad identity.
        condition: Filename-derived condition for MirrorGame.
        contrast: MirrorGame binary contrast.
        bad_segments: Circling bad six-second segment dictionary.
        fg_endpoint: strict requires 800 samples; trim permits removing only sample 801.

    Returns:
        Float32 microvolt windows and per-window provenance, before division by 100.
    """
    epochs = epochs.copy()
    windows, rows = [], []
    if dataset == "mg":
        epochs.set_eeg_reference("average", projection=True, verbose=False)
        if epochs.info["sfreq"] != 200:
            epochs.resample(200, n_jobs=1, verbose=False)
        ordered(epochs)
        data = epochs.get_data(units="uV")
        if data.shape[-1] != 1200:
            raise ValueError(f"MirrorGame requires 1200 resampled samples; got {data.shape[-1]}.")
        for trial, signal in enumerate(data):
            for start in (0, 400):
                windows.append(signal[:, start : start + 800])
                rows.append(
                    dict(
                        source_epoch=int(epochs.selection[trial]),
                        start_sample=start,
                        source_sfreq=200,
                        condition=condition,
                        target=CONTRASTS[contrast][condition],
                    )
                )
    elif dataset == "fg":
        if not np.isclose(epochs.tmin, -0.5, atol=1 / epochs.info["sfreq"]):
            raise ValueError(f"Expected FG start at -0.5 seconds; got {epochs.tmin}.")
        epochs.crop(tmax=3.5)
        if epochs.info["sfreq"] != 200:
            epochs.resample(200, n_jobs=1, verbose=False)
        ordered(epochs)
        data = epochs.get_data(units="uV")
        samples = data.shape[-1]
        if samples == 801 and fg_endpoint == "trim":
            data = data[..., :800]
        elif samples != 800:
            raise ValueError(
                f"FG crop produced {samples} samples, not 800. For 801 only, opt in with --fg-endpoint trim."
            )
        events = {value: key for key, value in epochs.event_id.items()}
        for trial, signal in enumerate(data):
            name = events[int(epochs.events[trial, 2])]
            windows.append(signal)
            rows.append(
                dict(
                    source_epoch=int(epochs.selection[trial]),
                    start_sample=0,
                    source_sfreq=200,
                    condition=name,
                    target=int(not name.endswith("n")),
                )
            )
    else:
        epochs.crop(tmin=2, verbose=False)
        epochs = epochs[12 if group == 2030 else 6 :]
        sfreq = epochs.info["sfreq"]
        length, stride = int(4 * sfreq), int(2 * sfreq)
        for name, target in {"open": 1, "closed1": 0}.items():
            if name not in epochs.event_id:
                continue
            selected = epochs[name]
            bad = bad_segments.get(f"{group}_{name}", [])
            for trial, signal in enumerate(selected.get_data(units="uV")):
                ranges = [(int(index) % 3 * 6, (int(index) % 3 + 1) * 6) for index in bad if int(index) // 3 == trial]
                for start in range(0, signal.shape[-1] - length + 1, stride):
                    if any(start / sfreq < end and (start + length) / sfreq > begin for begin, end in ranges):
                        continue
                    window = mne.EpochsArray(signal[None, :, start : start + length], selected.info, verbose=False)
                    if sfreq != 200:
                        window.resample(200, n_jobs=1, verbose=False)
                    ordered(window)
                    windows.append(window.get_data()[0])
                    rows.append(
                        dict(
                            source_epoch=int(selected.selection[trial]),
                            condition_trial=trial,
                            start_sample=start,
                            source_sfreq=sfreq,
                            condition=name,
                            target=target,
                        )
                    )
    if not windows:
        raise ValueError("No usable windows remain in this file.")
    array = np.asarray(windows, dtype=np.float32)
    if array.shape[1:] != (64, 800) or not np.isfinite(array).all():
        raise ValueError(f"Invalid window data: {array.shape}; expected finite (windows, 64, 800).")
    return array, pd.DataFrame(rows)


def prepare_cache(
    directory: Path,
    destination: Path,
    dataset: str,
    contrast: str = "solo-coordination",
    bad_path: Path | None = None,
    fg_endpoint: str = "strict",
) -> None:
    """Prepare one array per source file, publishing a cache marker only after success."""
    files = discover(directory, dataset, contrast)
    bad = {}
    if dataset == "circling":
        bad_path = bad_path or directory.parent / "dict_with_short_bad_epochs.pickle"
        with bad_path.open("rb") as stream:
            bad = pickle.load(stream)
        if not isinstance(bad, dict):
            raise ValueError("Circling bad segments must be a dictionary.")
    destination.mkdir(parents=True, exist_ok=False)
    files.to_csv(destination / "files.csv", index=False)
    all_metadata = []
    for index, row in enumerate(files.to_dict("records")):
        path = Path(row["source"])
        print(f"[{index + 1}/{len(files)}] {path.name}", flush=True)
        epochs = (
            mne.io.read_epochs_eeglab(path, verbose=False)
            if dataset == "mg"
            else mne.read_epochs(path, preload=True, verbose=False)
        )
        signal, metadata = extract_windows(epochs, dataset, row["group"], row["condition"], contrast, bad, fg_endpoint)
        filename = f"recording_{index:04d}.npy"
        np.save(destination / filename, signal, allow_pickle=False)
        for key in ("source", "group", "participant", "fold"):
            metadata[key] = row[key]
        metadata["array"] = filename
        metadata["array_index"] = np.arange(len(metadata))
        metadata["source_bytes"] = path.stat().st_size
        metadata["source_mtime_ns"] = path.stat().st_mtime_ns
        all_metadata.append(metadata)
    metadata = pd.concat(all_metadata, ignore_index=True)
    metadata.to_csv(destination / "windows.csv", index=False)
    config = dict(
        format_version=1,
        dataset=dataset,
        contrast=contrast if dataset == "mg" else "closed-open" if dataset == "circling" else "feedback",
        channels=CHANNELS,
        sfreq=200,
        n_times=800,
        stored_units="microvolts",
        model_divisor=100,
        fold_seed=42,
        fg_endpoint=fg_endpoint,
        n_windows=len(metadata),
        bad_segments_sha256=sha256(bad_path) if bad_path else None,
        metadata_sha256=sha256(destination / "windows.csv"),
    )
    (destination / "cache.json").write_text(json.dumps(config, indent=2) + "\n")


class WindowDataset(Dataset):
    """A bounded-memory adapter exposing cached windows to EEGClassifier.

    Arrays are reopened as read-only memory maps per access; worker processes do
    not inherit open handles. The OS page cache serves subsequent reads.
    """

    def __init__(self, cache: Path, metadata: pd.DataFrame) -> None:
        """Retain only metadata in RAM; store paths relative to the cache for portability."""
        self.cache = cache
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, list[int]]:
        """Return scaled float32 signal, binary label, and Braindecode crop indices."""
        row = self.metadata.iloc[index]
        path = (self.cache / row["array"]).resolve()
        if not path.is_relative_to(self.cache.resolve()):
            raise ValueError("Cache array path escapes the cache directory.")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        signal = np.array(array[int(row.array_index)], dtype=np.float32) / 100
        return signal, int(row.target), [0, 0, 800]


def fold_datasets(cache: Path, fold: int) -> tuple[dict, dict[str, WindowDataset]]:
    """Load complete-cache metadata and assert group and participant disjointness."""
    if not 0 <= fold < 10:
        raise ValueError("Fold must be in 0..9.")
    config = json.loads((cache / "cache.json").read_text())
    if config.get("format_version") != 1 or config.get("channels") != CHANNELS:
        raise ValueError("Unsupported cache format or channel order.")
    if sha256(cache / "windows.csv") != config["metadata_sha256"]:
        raise ValueError("Cache metadata changed after preparation.")
    metadata = pd.read_csv(cache / "windows.csv", dtype={"participant": str})
    if metadata.groupby("group").fold.nunique().max() != 1 or not set(metadata.target) <= {0, 1}:
        raise ValueError("Invalid group assignment or labels in cache.")
    metadata["split"] = np.where(
        metadata.fold == fold, "test", np.where(metadata.fold == (fold + 1) % 10, "valid", "train")
    )
    result = {}
    for role in ("train", "valid", "test"):
        rows = metadata[metadata.split == role]
        if rows.empty or set(rows.target) != {0, 1}:
            raise ValueError(f"{role} fold must contain both classes; inspect group/label counts.")
        result[role] = WindowDataset(cache, rows)
    for key in ("group", "participant"):
        sets = [set(result[role].metadata[key]) for role in result]
        if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"{key} leakage between partitions.")
    return config, result
