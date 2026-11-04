"""Regenerate the three movement intermediates from the Polhemus CSVs."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import movement_data_analysis_functions as md
from .common import PAIRS, dataset_root, save_pickle

PAIR2020_SHA256 = "f4f560ed82b86fcc20c3cc496181452a0508e253294f8dc9e76b68e35342c941"


def trial_phase(samples: dict, trials: int, onsets: np.ndarray | None) -> list[np.ndarray]:
    """Split one sensor into conditions and extract Hilbert phase from vertical position."""
    frame = pd.DataFrame(samples)
    if onsets is None:
        conditions = md.return_by_condition(frame, trials)
    else:
        conditions = [[frame.iloc[i : i + 4800] for i in onsets[c::3]] for c in range(3)]
    for segments in conditions:
        if len(segments) != trials or any(len(segment) != 4800 for segment in segments):
            raise ValueError(f"Expected {trials} complete 20-second movement trials per condition.")
    phase = [md.zscore_filter_phase_1d(segments, 10) for segments in conditions]
    if not np.isfinite(phase).all():
        raise ValueError("Nonfinite movement phase.")
    return phase


def pair_movement(root: Path, pair: int) -> tuple:
    """Reproduce both sensor phases and the notebook's first-20-trial synchronization scores."""
    directory = root / "behaviour_signal" / f"pair{pair}"
    files = sorted(directory.glob("polhemus*.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected one Polhemus CSV in {directory}, found {files}.")
    onsets = None
    if pair == 2020:
        with files[0].open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != PAIR2020_SHA256:
            raise ValueError("Pair 2020 CSV differs from the recording verified against the reconstructed onsets.")
        onsets = np.loadtxt(Path(__file__).with_name("pair2020_onsets.txt"), dtype=int)
    frame = (
        pd.read_csv(files[0])
        .rename(
            columns={
                "Timestamp": "Marker",
                "Marker": "Position X",
                "Position X": "Position Y",
                "Position Y": "Position Z",
                "Position Z": "Timestamp",
            }
        )
        .fillna({"Timestamp": 0})
    )
    trials = 21 if pair == 2450 else 20
    phase = tuple(trial_phase(sensor, trials, onsets) for sensor in md.split_interpolate(frame))
    ccor, plv = [], []
    for first, second in zip(*phase, strict=True):
        ccor.append([abs(md.ccorr_uniform(a, b)) for a, b in zip(first[:20], second[:20], strict=True)])
        plv.append([md.plv_custom(a, b) for a, b in zip(first[:20], second[:20], strict=True)])
    return phase, ccor, plv


def main() -> None:
    """Run the movement notebook's data-producing stage for the original ordered cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Override CIRCLING_DATASET_PATH.")
    parser.add_argument(
        "--report", action="store_true", help="Also save synchronization summaries and permutation test."
    )
    parser.add_argument("--output", type=Path, required=True, help="New movement output directory.")
    args = parser.parse_args()
    root = dataset_root(args.data_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    phases, ccors, plvs = [], [], []
    for pair in PAIRS:
        print(f"Processing movement for pair {pair}", flush=True)
        phase, ccor, plv = pair_movement(root, pair)
        phases.append(phase)
        ccors.append(ccor)
        plvs.append(plv)
    for name, data in (("phase_list", phases), ("ccor_results", ccors), ("plv_results", plvs)):
        save_pickle(args.output / f"{name}.pickle", data)

    if args.report:
        from .reports import movement_report

        movement_report(args.output, phases, ccors, plvs)


if __name__ == "__main__":
    main()
