"""Lesson 2: MNE windows, selectable contrasts, and dyad-disjoint datasets."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset, create_from_mne_epochs
from sklearn.model_selection import KFold

if __package__:
    from . import inspect_mirrorgame as preprocessing
else:
    import inspect_mirrorgame as preprocessing


def make_manifest(directory: Path, contrast: str, fold: int = 0) -> pd.DataFrame:
    """Assign files to the original seeded ten-fold dyad protocol.

    Args:
        directory: Directory containing preprocessed SET files.
        contrast: Key from preprocessing.CONTRASTS.
        fold: Test fold, from zero through nine.

    Returns:
        File table with labels, identities, fold IDs, and train/valid/test roles.

    Raises:
        ValueError: If the contrast, fold, group count, or file coverage is invalid.
    """
    if contrast not in preprocessing.CONTRASTS or not 0 <= fold < 10:
        raise ValueError("Select a supported contrast and a fold from 0 to 9.")
    labels = preprocessing.CONTRASTS[contrast]
    rows = []
    for path in sorted(directory.glob("*.set")):
        participant, dyad, condition = preprocessing.recording_identity(path)
        if condition in labels:
            rows.append(dict(path=str(path), participant=participant, dyad=dyad,
                             condition=condition, target=labels[condition]))
    if not rows:
        raise ValueError("No files found for the selected contrast.")
    manifest = pd.DataFrame(rows)
    for dyad, group in manifest.groupby("dyad"):
        expected = {(f"{member}{dyad:02d}", condition) for member in (1, 2) for condition in labels}
        actual = set(zip(group.participant, group.condition))
        if actual != expected or len(group) != 4:
            raise ValueError(f"Dyad {dyad} does not have both participants and both selected conditions.")
    dyads = np.array(sorted(manifest.dyad.unique()))
    if len(dyads) < 10:
        raise ValueError("The original ten-fold protocol requires at least ten dyads.")
    np.random.RandomState(42).shuffle(dyads)
    assignments = {}
    for index, (_, test_indices) in enumerate(KFold(n_splits=10).split(dyads)):
        assignments.update({int(dyad): index for dyad in dyads[test_indices]})
    manifest["fold"] = manifest.dyad.map(assignments)
    manifest["split"] = np.where(manifest.fold == fold, "test",
                                 np.where(manifest.fold == (fold + 1) % 10, "valid", "train"))
    assert_disjoint(manifest)
    return manifest


def assert_disjoint(metadata: pd.DataFrame) -> None:
    """Reject participant or dyad overlap between dataset partitions.

    Args:
        metadata: Table containing split, dyad, and participant columns.

    Raises:
        ValueError: If any identity occurs in two partitions.
    """
    groups = dict(tuple(metadata.groupby("split")))
    for left, right in combinations(groups, 2):
        for column in ("dyad", "participant"):
            overlap = set(groups[left][column]) & set(groups[right][column])
            if overlap:
                raise ValueError(f"{column} leakage between {left} and {right}: {overlap}")


def scale_for_labram(signal: np.ndarray) -> np.ndarray:
    """Convert a float32 window in volts to the original microvolts/100 scale.

    Args:
        signal: Window supplied by Braindecode in volts.

    Returns:
        Float32 model input; rounding may differ slightly from casting microvolts first.
    """
    return signal * np.float32(1e6) / np.float32(100)


def windows_from_epochs(epochs: mne.BaseEpochs, description: dict) -> BaseConcatDataset:
    """Wrap six-second epochs without applying pending reference projectors.

    Args:
        epochs: Prepared epochs, with 1200 samples at 200 Hz.
        description: File metadata including the binary target and partition.

    Returns:
        Braindecode dataset with two windows per epoch and explicit provenance.

    Raises:
        ValueError: If timing, signal values, or the target are invalid.
    """
    if epochs.info["sfreq"] != preprocessing.SFREQ or len(epochs.times) != 1200:
        raise ValueError("Expected six-second epochs at 200 Hz.")
    if description["target"] not in (0, 1) or not np.isfinite(epochs.get_data()).all():
        raise ValueError("Expected a binary target and finite EEG values.")
    safe_epochs = epochs.copy()
    inactive = [i for i, projector in enumerate(safe_epochs.info["projs"]) if not projector["active"]]
    if inactive:
        safe_epochs.del_proj(inactive)
    mapping = {int(code): int(description["target"]) for code in np.unique(epochs.events[:, 2])}
    with mne.utils.use_log_level("ERROR"):
        dataset = create_from_mne_epochs(
            [safe_epochs], window_size_samples=preprocessing.WINDOW_SAMPLES, window_stride_samples=400,
            drop_last_window=True, descriptions=[description], mapping=mapping, preload=True,
        )
        for trial, child in enumerate(dataset.datasets):
            if len(child) != 2:
                raise ValueError(f"Expected two windows for source epoch {trial}.")
            child.set_description({"source_epoch": int(epochs.selection[trial])})
            metadata = child.windows.metadata.copy()
            metadata["start_sample"] = [0, 400]
            metadata["stop_sample"] = [800, 1200]
            metadata["start_time_seconds"] = epochs.tmin + np.array([0, 400]) / preprocessing.SFREQ
            child.windows.metadata = metadata
            child.transform = scale_for_labram
    return dataset


def build_datasets(manifest: pd.DataFrame) -> dict[str, BaseConcatDataset]:
    """Load the listed recordings and split their windows by assigned role.

    Args:
        manifest: File table produced by make_manifest, optionally subset for preview.

    Returns:
        Train, valid, and test datasets for the roles present in the manifest.
    """
    assert_disjoint(manifest)
    children = []
    for row in manifest.to_dict("records"):
        path = Path(row["path"])
        print(f"Loading {path.name} [{row['split']}]", flush=True)
        dataset = windows_from_epochs(preprocessing.prepare_epochs(path), row)
        children.extend(dataset.datasets)
    combined = BaseConcatDataset(children)
    assert_disjoint(combined.description)
    return combined.split("split")


def main() -> None:
    """Preview one dyad per partition or load the full selected contrast."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", choices=preprocessing.CONTRASTS, default=preprocessing.DEFAULT_CONTRAST)
    parser.add_argument("--fold", type=int, choices=range(10), default=0)
    parser.add_argument("--all", action="store_true", help="Load all selected files; requires several GB of RAM.")
    parser.add_argument("--manifest-only", action="store_true", help="Inspect assignments without loading EEG.")
    args = parser.parse_args()
    manifest = make_manifest(preprocessing.get_data_directory(), args.contrast, args.fold)
    print(f"Contrast: {preprocessing.CONTRASTS[args.contrast]}; test fold: {args.fold}")
    for name, group in manifest.groupby("split"):
        print(f"{name}: {group.dyad.nunique()} dyads, {group.participant.nunique()} participants, {len(group)} files")
        print(f"  dyads: {sorted(group.dyad.unique().tolist())}")
    if args.manifest_only:
        return
    if not args.all:
        chosen = manifest.groupby("split").dyad.min().tolist()
        manifest = manifest[manifest.dyad.isin(chosen)]
        print("PREVIEW ONLY: one complete dyad per partition; full fold assignments remain unchanged.")
    datasets = build_datasets(manifest)
    for name, dataset in datasets.items():
        metadata = dataset.get_metadata()
        signal, target, crop_indices = dataset[0]
        print(f"{name}: {len(dataset)} windows; class counts: {metadata.target.value_counts().to_dict()}")
        print(f"  first item: X={signal.shape}, dtype={signal.dtype}, y={target}, crop indices={crop_indices}")
        columns = ["participant", "dyad", "source_epoch", "start_sample", "target"]
        print(metadata[columns].head(2).to_string(index=False))


if __name__ == "__main__":
    main()
