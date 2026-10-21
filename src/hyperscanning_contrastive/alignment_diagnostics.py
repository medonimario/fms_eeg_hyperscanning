"""Fixed-gallery and optimizer diagnostics for frozen-feature partner alignment."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional

from .contrastive import contrastive_loss, pair_scores
from .ssl import gallery_metrics


def embedding_geometry(values: torch.Tensor) -> dict[str, float]:
    """Measure within-person normalized spread and entropy rank of centered embeddings.

    Effective rank is bounded by min(number of gallery trials minus one, feature dimension).
    Exactly constant embeddings return rank zero. Small spread must be interpreted alongside rank.
    """
    normalized = functional.normalize(values, dim=1)
    centered = normalized - normalized.mean(0)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    rank = 0.0
    if total > 1e-7:
        probability = singular / total
        rank = (-(probability * probability.clamp_min(1e-12).log()).sum()).exp().item()
    return {
        "normalized_std": normalized.std(0, unbiased=False).mean().item(),
        "effective_rank": rank,
        "embedding_norm": values.norm(dim=1).mean().item(),
    }


def fixed_diagnostics(
    head: nn.Module, bank: dict, galleries: list[pd.DataFrame], mappings: dict | None = None
) -> pd.DataFrame:
    """Evaluate fixed true or assigned-training matches without changing training RNGs."""
    head.eval()
    rows = []
    with torch.no_grad():
        for frame in galleries:
            key = (int(frame.group.iloc[0]), frame.condition.iloc[0])
            entry = bank[key]
            lookup = {trial: i for i, trial in enumerate(entry["rows"].trial_id)}
            indices = np.array([lookup[trial] for trial in frame.trial_id])
            partner_indices = indices if mappings is None else mappings[key][indices]
            a, b = head(entry["a"][indices]), head(entry["b"][partner_indices])
            scores = pair_scores(a, b)
            positive = scores.diagonal().mean().item()
            negative = scores[~torch.eye(len(scores), dtype=torch.bool)].mean().item()
            geometry_a, geometry_b = embedding_geometry(a), embedding_geometry(b)
            geometry = {name: (value + geometry_b[name]) / 2 for name, value in geometry_a.items()}
            rows.append(
                {
                    "group": key[0],
                    "condition": key[1],
                    "loss": contrastive_loss(a, b).item(),
                    "positive_cosine": positive,
                    "negative_cosine": negative,
                    "cosine_margin": positive - negative,
                    **gallery_metrics(a, b),
                    **geometry,
                }
            )
    return pd.DataFrame(rows)


def record_diagnostics(
    head: nn.Module,
    bank: dict,
    training: list[pd.DataFrame],
    validation: list[pd.DataFrame],
    mappings: dict | None,
    epoch: int,
    folder: Path,
    validation_bank: dict | None = None,
) -> dict[str, float]:
    """Append per-gallery diagnostics and return equal-condition/dyad macro metrics for history."""
    result, records = {}, []
    for name, galleries, assignment in (
        ("train_true", training, None),
        ("train_objective", training, mappings),
        ("validation", validation, None),
    ):
        selected_bank = validation_bank if name == "validation" and validation_bank is not None else bank
        frame = fixed_diagnostics(head, selected_bank, galleries, assignment)
        columns = frame.columns.difference(["group", "condition"])
        means = frame.groupby(["group", "condition"])[columns].mean().groupby("group").mean().mean()
        result.update({f"{name}_{key}": float(value) for key, value in means.items()})
        records.append(frame.assign(epoch=epoch, evaluation=name))
    pd.concat(records).to_csv(
        folder / "diagnostic_galleries.csv", index=False, mode="w" if epoch == 0 else "a", header=epoch == 0
    )
    return result
