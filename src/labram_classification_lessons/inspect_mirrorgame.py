"""Lesson 1: inspect MirrorGame and reproduce its two windows per epoch."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path

import mne
import numpy as np
from dotenv import load_dotenv

STANDARD_CHANNELS = (
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
DEFAULT_CONTRAST = "solo-coordination"
SFREQ = 200
WINDOW_STARTS = (0, 400)
WINDOW_SAMPLES = 800


def get_data_directory() -> Path:
    """Resolve MG_DATASET_PATH from the environment or repository .env.

    Returns:
        Directory containing preprocessed EEGLAB files.

    Raises:
        ValueError: If the variable is unset.
        FileNotFoundError: If neither supported directory contains SET files.
    """
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    value = os.environ.get("MG_DATASET_PATH")
    if not value:
        raise ValueError("Set MG_DATASET_PATH in the repository .env or environment.")
    root = Path(value).expanduser()
    for candidate in (root, root / "preprocessed"):
        if candidate.is_dir() and any(candidate.glob("*.set")):
            return candidate
    raise FileNotFoundError(f"No SET files found in {root} or its preprocessed subdirectory.")


def recording_identity(path: Path) -> tuple[str, int, str]:
    """Extract participant, dyad, and condition using the original naming rule.

    Args:
        path: Filename such as s_103_Solo.set.

    Returns:
        Participant code, dyad number, and condition name.

    Raises:
        ValueError: If the filename does not follow the expected convention.
    """
    match = re.fullmatch(r"s_([12])(\d{2})_(Solo|Spontaneous|Coordination)\.set", path.name)
    if match is None:
        raise ValueError(f"Unexpected MirrorGame filename: {path.name}")
    member, dyad, condition = match.groups()
    return f"{member}{dyad}", int(dyad), condition


def prepare_epochs(path: Path) -> mne.BaseEpochs:
    """Repeat the original reference, resampling, and channel selection sequence.

    Args:
        path: Preprocessed EEGLAB epoch file.

    Returns:
        Epochs in volts, with 64 ordered channels and 1200 samples at 200 Hz.

    Raises:
        ValueError: If required channels or the expected epoch length are absent.
    """
    epochs = mne.io.read_epochs_eeglab(path, verbose=False)
    epochs.set_eeg_reference("average", projection=True, verbose=False)
    if epochs.info["sfreq"] != SFREQ:
        epochs.resample(SFREQ, n_jobs=1, verbose=False)
    missing = sorted(set(STANDARD_CHANNELS) - set(epochs.ch_names))
    if missing:
        raise ValueError(f"Missing standard channels in {path.name}: {missing}")
    epochs.reorder_channels(STANDARD_CHANNELS)
    if len(epochs.times) != 1200:
        raise ValueError(f"Expected 1200 samples after resampling; got {len(epochs.times)} in {path.name}.")
    return epochs


def make_windows(epochs: mne.BaseEpochs) -> np.ndarray:
    """Extract the original two overlapping windows, retaining the epoch axis.

    Args:
        epochs: Prepared six-second MNE epochs at 200 Hz.

    Returns:
        Microvolt array shaped (epochs, 2, channels, 800), before division by 100.

    Raises:
        ValueError: If input timing or signal values are invalid.
    """
    if epochs.info["sfreq"] != SFREQ or len(epochs.times) != 1200:
        raise ValueError("Windowing requires 1200-sample epochs at 200 Hz.")
    data = epochs.get_data(units="uV")
    if not np.isfinite(data).all():
        raise ValueError("EEG contains non-finite values.")
    return np.stack([data[..., start : start + WINDOW_SAMPLES] for start in WINDOW_STARTS], axis=1)


def main() -> None:
    """Inventory files and inspect windows from one selected recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Filename to inspect; defaults to the first class-0 file.")
    parser.add_argument("--contrast", choices=CONTRASTS, default=DEFAULT_CONTRAST)
    args = parser.parse_args()
    labels = CONTRASTS[args.contrast]
    directory = get_data_directory()
    files = sorted(directory.glob("*.set"))
    identities = {path: recording_identity(path) for path in files}
    selected = [path for path in files if identities[path][2] in labels]
    path = directory / args.file if args.file else next(p for p in selected if labels[identities[p][2]] == 0)
    if path not in identities or identities[path][2] not in labels:
        raise ValueError("Choose a filename belonging to the selected contrast in MG_DATASET_PATH.")
    participant, dyad, condition = identities[path]
    print(f"MNE version: {mne.__version__}")
    print(f"Files by condition: {dict(Counter(identity[2] for identity in identities.values()))}")
    print(f"Selected contrast: {labels}; {len(selected)} files")
    print(f"Participants: {len({identities[p][0] for p in selected})}")
    print(f"Dyads: {len({identities[p][1] for p in selected})}")
    print(f"Inspecting {path.name}: participant={participant}, dyad={dyad}, label={labels[condition]}")
    epochs = prepare_epochs(path)
    windows = make_windows(epochs)
    print(f"Prepared epochs: {len(epochs)}; sampling rate: {epochs.info['sfreq']} Hz")
    print(f"Sample timestamps: {epochs.times[0]} to {epochs.times[-1]} seconds")
    print(f"Event IDs: {epochs.event_id}")
    print(f"Ordered channels: {epochs.ch_names}")
    print(f"Projectors (description, active): {[(p['desc'], p['active']) for p in epochs.info['projs']]}")
    print(f"Window shape (epochs, windows, channels, samples): {windows.shape}")
    print(f"Number of training examples: {windows.shape[0] * windows.shape[1]}")
    print(f"Window slices: [0:800], [400:1200]; input scaling later: microvolts / 100")
    np.testing.assert_array_equal(windows[:, 0, :, 400:], windows[:, 1, :, :400])
    print("Verified: the shared 2 seconds are identical. Both windows retain the same dyad identity.")


if __name__ == "__main__":
    main()
