"""Fit Circling forward mTRFs and collect the participant-level notebook results."""

import argparse
import pickle
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, zscore

from .common import CONDITIONS, PAIRS, dataset_root, save_pickle
from .movement_data_analysis_functions import new_rel_phase
from .mtrf_modified import TRF
from .mtrf_modified.stats import nested_crossval

MODELS = {
    "Full": (0, 1, 2, 3, 4),
    "SelfOther": (0, 1, 2, 3),
    "SelfCoordination": (0, 1, 4),
    "OtherCoordination": (2, 3, 4),
}
REDUCED = {"self": "OtherCoordination", "other": "SelfCoordination", "coordination": "SelfOther"}
LABELS = dict(zip(CONDITIONS, ("Before", "Vision", "After"), strict=True))
FS = 240
LAGS = (-0.250, 0.300)
LAMBDAS = np.logspace(-1, 6, 10)
CV_FILE = "trf_long_all_rsq_sin_cos_2.pickle"
MAP_FILE = "trf_visualization_sin_cos_2.pickle"


def eeg_path(root: Path, pair: int, member: str) -> Path:
    """Resolve the flat participant FIF without guessing A/B from file order."""
    directory = root / "preprocessed"
    paths = [path for sep in ("", "_", "-") if (path := directory / f"pair{pair}{sep}{member}_ave_epo.fif").is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected one pair{pair} member {member} *_ave_epo.fif in {directory}, found {paths}.")
    return paths[0]


def retained_trials(bad: list[int]) -> list[int]:
    """Replicate segment_to_epoch and the notebook's literal membership test against range(20)."""
    excluded = {(int(index) - 1) % 20 + 1 for index in bad}
    return [i for i in range(20) if i not in excluded]


def movement_phase(phase: tuple, pair: int) -> np.ndarray:
    """Remove the extra movement trial from pair 2450 and crop two seconds for EEG alignment."""
    values = np.asarray(phase)
    expected = (2, 3, 21 if pair == 2450 else 20, 4800)
    if values.shape != expected:
        raise ValueError(f"Pair {pair} phase shape {values.shape}, expected {expected}.")
    if pair == 2450:
        values = np.delete(values, 10, axis=2)
    return values[..., 480:]


def electrode_scores(stim: list, resp: list, regularization: float) -> np.ndarray:
    """Replicate the Vision topography refit on all retained trials with the median fold penalty."""
    trf = TRF(direction=1)
    trf.train(stim, resp, FS, *LAGS, regularization)
    observed = np.asarray(resp)
    residuals = observed - np.asarray(trf.predict(stim))
    ss_res = (residuals**2).sum(axis=1)
    ss_tot = ((observed - observed.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return (1 - ss_res / ss_tot).mean(axis=0)


def fit_pair(root: Path, pair: int, phase: tuple, bad: dict, jobs: int) -> tuple[dict, dict]:
    """Fit the four notebook models for both participants under all three conditions."""
    phase = movement_phase(phase, pair)
    epochs = [
        mne.read_epochs(eeg_path(root, pair, member), preload=True, verbose=False).crop(tmin=2)[
            12 if pair == 2030 else 6 :
        ]
        for member in ("a", "b")
    ]
    if epochs[0].ch_names != epochs[1].ch_names:
        raise ValueError(f"Pair {pair} has mismatched EEG channel order.")
    scores = {f"{c}_{model}": [] for c in CONDITIONS for model in MODELS}
    maps = {f"open_{effect}": [] for effect in ("full", *REDUCED)}
    for c, condition in enumerate(CONDITIONS):
        keep = retained_trials(bad[f"{pair}_{condition}"])
        first, second = (e[condition] for e in epochs)
        if len(first) != 20 or len(second) != 20 or not np.array_equal(first.selection, second.selection):
            raise ValueError(f"Pair {pair} {condition} needs 20 aligned trials per participant after practice removal.")
        if not np.array_equal(first.events[:, 0], second.events[:, 0]):
            raise ValueError(f"Pair {pair} {condition} EEG event times differ.")
        a, b = phase[:, c][:, keep]
        coord = np.cos(new_rel_phase(a, b))
        for selected, own, other in ((first, a, b), (second, b, a)):
            data = zscore(selected[keep].resample(FS, verbose=False).get_data(), axis=-1).transpose(0, 2, 1)
            if data.shape[:2] != own.shape or not np.isfinite(data).all():
                raise ValueError(
                    f"Pair {pair} {condition} has misaligned or nonfinite EEG: {data.shape} vs {own.shape}."
                )
            features = (np.cos(own), np.sin(own), np.cos(other), np.sin(other), coord)
            topo = {}
            for model, columns in MODELS.items():
                stim = list(np.stack([features[i] for i in columns], axis=-1))
                resp = list(data)
                result = nested_crossval(
                    TRF(direction=1, preload=False),
                    stim,
                    resp,
                    FS,
                    *LAGS,
                    LAMBDAS,
                    k=5,
                    seed=123,
                    n_jobs=jobs,
                    verbose=False,
                )
                if not all(np.isfinite(values).all() for values in result):
                    raise ValueError(f"Nonfinite fold result for pair {pair}, condition {condition}, model {model}.")
                scores[f"{condition}_{model}"].append(result)
                if condition == "open":
                    topo[model] = electrode_scores(stim, resp, np.median(result[1]))
            if topo:
                maps["open_full"].append([topo["Full"]])
                for effect, reduced in REDUCED.items():
                    maps[f"open_{effect}"].append([topo["Full"] - topo[reduced]])
    return scores, maps


def collect(directory: Path, root: Path | None = None) -> None:
    """Merge the complete cohort in notebook order and reproduce its ΔR² summaries."""
    scores, maps = {}, {}
    for pair in PAIRS:
        with (directory / f"pair{pair}.pickle").open("rb") as stream:
            pair_scores, pair_maps = pickle.load(stream)
        for merged, shard in ((scores, pair_scores), (maps, pair_maps)):
            for key, participants in shard.items():
                merged.setdefault(key, []).extend(participants)
    save_pickle(directory / CV_FILE, scores)
    save_pickle(directory / MAP_FILE, maps)
    rows = []
    for c in CONDITIONS:
        averaged = {m: np.tanh(np.arctanh([v[2] for v in scores[f"{c}_{m}"]]).mean(axis=1)) for m in MODELS}
        for idx, pair in enumerate(PAIRS):
            if pair == 2020:
                continue
            for member, label in enumerate(("a", "b")):
                i = idx * 2 + member
                for effect, reduced in REDUCED.items():
                    rows.append((pair, label, LABELS[c], effect, averaged["Full"][i] - averaged[reduced][i]))
    df = pd.DataFrame(rows, columns=["pair", "member", "condition", "variable", "delta_r2"])
    df.to_csv(directory / "delta_r2.csv", index=False)
    summary = df.groupby(["condition", "variable"], sort=False).delta_r2.agg(
        median="median", wilcoxon_p=lambda x: wilcoxon(x).pvalue
    )
    summary.to_csv(directory / "delta_r2_summary.csv")
    contrasts = []
    for effect, frame in df.groupby("variable", sort=False):
        wide = frame.pivot(index=["pair", "member"], columns="condition", values="delta_r2")
        for a, b in (("Vision", "Before"), ("Vision", "After"), ("Before", "After")):
            contrasts.append((effect, a, b, 3 * wilcoxon(wide[a], wide[b]).pvalue))
    pd.DataFrame(contrasts, columns=["variable", "first", "second", "notebook_p_times_three"]).to_csv(
        directory / "condition_contrasts.csv", index=False
    )
    if root is not None:
        from .reports import eeg_figures

        info = mne.io.read_info(eeg_path(root, PAIRS[0], "a"), verbose=False)
        eeg_figures(directory, df, maps, info)


def main() -> None:
    """Fit one array task or the full cohort, or merge the completed array."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--data-dir", type=Path, help="Override CIRCLING_DATASET_PATH.")
    fit.add_argument("--phase", type=Path, required=True, help="Trusted original or regenerated phase_list.pickle.")
    fit.add_argument("--bad-segments", type=Path, help="Defaults to dict_with_short_bad_epochs.pickle in dataset root.")
    fit.add_argument("--output", type=Path, required=True, help="Separate run directory for all dyad shards.")
    fit.add_argument("--jobs", type=int, default=1, help="Number of concurrent outer folds.")
    group = fit.add_mutually_exclusive_group()
    group.add_argument("--pair", type=int, choices=PAIRS)
    group.add_argument("--pair-index", type=int, choices=range(1, len(PAIRS) + 1), help="One-based array index.")
    merge = commands.add_parser("collect")
    merge.add_argument("--data-dir", type=Path, help="Override CIRCLING_DATASET_PATH for scalp coordinates.")
    merge.add_argument("--output", type=Path, required=True, help="Run directory containing the 52 pair files.")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output, dataset_root(args.data_dir))
        return
    root = dataset_root(args.data_dir)
    with (args.bad_segments or root / "dict_with_short_bad_epochs.pickle").open("rb") as stream:
        bad = pickle.load(stream)
    with args.phase.open("rb") as stream:
        phases = pickle.load(stream)
    if len(phases) != len(PAIRS):
        raise ValueError(f"Expected the original ordered {len(PAIRS)}-pair phase list, got {len(phases)}.")
    pair = PAIRS[args.pair_index - 1] if args.pair_index else args.pair
    args.output.mkdir(parents=True, exist_ok=True)
    for selected in (pair,) if pair else PAIRS:
        print(f"Fitting pair {selected}", flush=True)
        results = fit_pair(root, selected, phases[PAIRS.index(selected)], bad, args.jobs)
        save_pickle(args.output / f"pair{selected}.pickle", results)
    if pair is None:
        collect(args.output, root)


if __name__ == "__main__":
    main()
