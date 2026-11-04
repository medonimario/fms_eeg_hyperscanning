"""Headless group summaries for the Circling movement and EEG notebook outputs."""

from pathlib import Path

import matplotlib
import mne
import numpy as np
import pandas as pd
from scipy.stats import iqr, wilcoxon, zscore

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def movement_report(output: Path, phases: list, ccors: list, plvs: list) -> None:
    """Summarize synchronization and reproduce the Vision partner-permutation test."""
    conditions = ("Before", "Vision", "After")
    rows, tests = [], []
    median_plv = np.median(plvs, axis=-1)
    for label, values in (
        ("PLV", median_plv),
        ("PLV truncated", median_plv[zscore(median_plv[:, 1]) <= 1]),
        ("Absolute circular correlation", np.median(ccors, axis=-1)),
    ):
        fig, ax = plt.subplots()
        ax.plot(conditions, values.T, alpha=0.15, color="gray")
        ax.plot(conditions, np.median(values, axis=0), "o-", color="black")
        ax.set(ylabel=label, ylim=(0, 1))
        fig.savefig(output / f"{label.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)
        for c, scores in zip(conditions, values.T, strict=True):
            rows.append((label, c, len(scores), np.median(scores), iqr(scores)))
        for a, b in ((0, 1), (2, 1), (0, 2)):
            tests.append((label, conditions[a], conditions[b], 3 * wilcoxon(values[:, a], values[:, b]).pvalue))
    pd.DataFrame(rows, columns=["measure", "condition", "dyads", "median", "iqr"]).to_csv(
        output / "movement_summary.csv", index=False
    )
    pd.DataFrame(tests, columns=["measure", "first", "second", "notebook_p_times_three"]).to_csv(
        output / "movement_contrasts.csv", index=False
    )
    # The original permutation cell takes the first 20 Vision trials, including for pair 2450.
    unit_phase = np.exp(1j * np.asarray([[p[0][1][:20], p[1][1][:20]] for p in phases]))
    dyads, _, trials, samples = unit_phase.shape
    trial_plv = np.empty((trials, dyads, dyads))
    for t in range(trials):
        trial_plv[t] = np.abs(unit_phase[:, 0, t] @ unit_phase[:, 1, t].conj().T) / samples
    pair_medians = np.median(trial_plv, axis=0)
    observed = np.median(pair_medians.diagonal())
    rng = np.random.default_rng(123)
    perm = np.array([np.median(pair_medians[np.arange(dyads), rng.permutation(dyads)]) for _ in range(1000)])
    p = min(1, 2 * (1 + min(np.sum(perm >= observed), np.sum(perm <= observed))) / 1001)
    pd.DataFrame({"permuted_median_plv": perm}).to_csv(output / "vision_permutations.csv", index=False)
    pd.DataFrame([(observed, np.median(perm), p)], columns=["observed", "permutation_median", "p_two_sided"]).to_csv(
        output / "vision_permutation_test.csv", index=False
    )


def eeg_figures(directory: Path, df: pd.DataFrame, maps: dict, info: mne.Info) -> None:
    """Save the held-out group contrasts and separate full-data Vision scalp maps."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True, layout="constrained")
    effects = ("self", "other", "coordination")
    for effect, ax in zip(effects, axes, strict=True):
        frame = df[df.variable == effect]
        ax.boxplot([frame[frame.condition == c].delta_r2 for c in ("Before", "Vision", "After")])
        ax.set_xticks([1, 2, 3], ["Before", "Vision", "After"])
        ax.axhline(0, linewidth=0.7, color="gray")
        ax.set_title(effect)
    axes[0].set_ylabel("Held-out ΔR²")
    fig.savefig(directory / "delta_r2.png", dpi=180)
    plt.close(fig)
    # Preserve cell 19's literal participant slice, which differs from the main analysis exclusion.
    profiles = [np.mean(np.concatenate(maps[f"open_{v}"][:1] + maps[f"open_{v}"][3:]), axis=0) for v in effects]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for effect, values, ax in zip(effects, profiles, axes, strict=True):
        image, _ = mne.viz.plot_topomap(
            values, info, axes=ax, show=False, cmap="RdBu_r", vlim=(np.min(profiles), 0.0025)
        )
        ax.set_title(effect)
    fig.colorbar(image, ax=axes, label="Vision refit ΔR²")
    fig.savefig(directory / "vision_topographies.png", dpi=180)
    plt.close(fig)
