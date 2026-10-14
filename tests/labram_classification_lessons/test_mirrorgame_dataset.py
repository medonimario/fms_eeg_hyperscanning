"""Validate window parity, binary contrasts, and the legacy group split protocol."""

import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np
from labram_classification_lessons.inspect_mirrorgame import CONTRASTS, STANDARD_CHANNELS
from labram_classification_lessons.mirrorgame_dataset import assert_disjoint, make_manifest, windows_from_epochs
from sklearn.model_selection import KFold


class MirrorGameTests(unittest.TestCase):
    """Check scientific data invariants without requiring the external drive."""

    def test_windows_preserve_signal_and_provenance(self) -> None:
        """An inactive projector must not alter the original sample slices."""
        rng = np.random.RandomState(7)
        data = rng.normal(size=(2, 64, 1200)) * 1e-5 + 3e-5
        info = mne.create_info(STANDARD_CHANNELS, 200, ch_types="eeg")
        info.set_montage("biosemi64")
        epochs = mne.EpochsArray(data, info, tmin=-1, verbose=False)
        epochs.set_eeg_reference("average", projection=True, verbose=False)
        description = dict(target=1, participant="101", dyad=1, split="train", path="synthetic")
        dataset = windows_from_epochs(epochs, description)
        self.assertEqual(len(dataset), 4)
        expected_locations = np.array([channel["loc"] for channel in epochs.info["chs"]])
        for child in dataset.datasets:
            actual_locations = np.array([channel["loc"] for channel in child.windows.info["chs"]])
            np.testing.assert_array_equal(actual_locations, expected_locations)
        self.assertFalse(epochs.info["projs"][0]["active"])
        for trial in range(2):
            for window, start in enumerate((0, 400)):
                signal, label, _ = dataset[trial * 2 + window]
                expected = (data[trial, :, start:start + 800] * 1e6).astype(np.float32) / 100
                np.testing.assert_allclose(signal, expected, rtol=3e-7, atol=1e-8)
                self.assertEqual(label, 1)
        metadata = dataset.get_metadata()
        self.assertEqual(metadata.source_epoch.tolist(), [0, 0, 1, 1])
        self.assertEqual(metadata.start_sample.tolist(), [0, 400, 0, 400])
        self.assertEqual(metadata.start_time_seconds.tolist(), [-1, 1, -1, 1])
        self.assertEqual(set(metadata.dyad), {1})

    def test_all_contrasts_and_folds_match_legacy(self) -> None:
        """Every contrast uses the original shuffled dyad membership for every fold."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dyad in range(1, 41):
                for member in (1, 2):
                    for condition in ("Solo", "Spontaneous", "Coordination"):
                        (root / f"s_{member}{dyad:02d}_{condition}.set").touch()
            dyads = np.arange(1, 41)
            np.random.RandomState(42).shuffle(dyads)
            legacy = [set(dyads[index]) for _, index in KFold(10).split(dyads)]
            for contrast, labels in CONTRASTS.items():
                for fold in range(10):
                    manifest = make_manifest(root, contrast, fold)
                    self.assertEqual(set(manifest[manifest.split == "test"].dyad), legacy[fold])
                    self.assertEqual(set(manifest[manifest.split == "valid"].dyad), legacy[(fold + 1) % 10])
                    self.assertEqual(manifest.groupby("split").dyad.nunique().to_dict(),
                                     {"test": 4, "train": 32, "valid": 4})
                    self.assertEqual(dict(zip(manifest.condition, manifest.target)), labels)
                    assert_disjoint(manifest)
            leaked = manifest.copy()
            leaked.loc[0, "split"] = "valid" if leaked.loc[0, "split"] != "valid" else "test"
            with self.assertRaises(ValueError):
                assert_disjoint(leaked)
            (root / "s_101_Solo.set").unlink()
            with self.assertRaises(ValueError):
                make_manifest(root, "solo-coordination")


if __name__ == "__main__":
    unittest.main()
