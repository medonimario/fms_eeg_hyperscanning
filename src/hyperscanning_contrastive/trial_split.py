"""Deterministic whole-trial validation within the original training dyads."""

import numpy as np

from .ssl import EEGCache


def split_within_dyad(cache: EEGCache, validation_trials: int, pairs: int, seed: int) -> None:
    """Reserve trials per dyad/condition, leaving external validation and test dyads unused.

    Split rows describe paired trials, so both participant views inherit one assignment.
    Only offset-zero windows are eligible in this experiment; no other trial windows are read.
    """
    if validation_trials < pairs:
        raise ValueError("Validation trials must be at least the gallery size.")
    rows = cache.rows.copy()
    rows.loc[rows.split == "validation", "split"] = "external_validation"
    rng = np.random.default_rng(seed)
    for (group, condition), frame in rows[rows.split == "train"].groupby(["group", "condition"], sort=True):
        frame = frame.sort_values("condition_trial")
        if len(frame) - validation_trials < pairs:
            raise ValueError(
                f"Dyad {group}, {condition}: {len(frame)} eligible trials cannot supply "
                f"{validation_trials} validation trials and {pairs} training pairs."
            )
        selected = rng.choice(frame.index.to_numpy(), validation_trials, replace=False)
        rows.loc[selected, "split"] = "validation"
    strata = {name: [] for name in ("train", "validation", "test")}
    for (split, _, _), frame in rows.groupby(["split", "group", "condition"], sort=True):
        if split not in strata:
            continue
        for member in (0, 1):
            strata[split].append(frame.sort_values("condition_trial").assign(member=member))
    cache.rows = rows
    cache.strata = strata
