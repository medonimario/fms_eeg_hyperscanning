"""Self-contained cross-modal contrastive loss and tie-aware retrieval metrics."""

from __future__ import annotations

import math

import torch
from torch.nn import functional


def paired_logits(eeg: torch.Tensor, movement: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Compute cosine logits for one gallery of distinct paired trials.

    Args:
        eeg: Query embeddings shaped (trials, features).
        movement: Candidate embeddings of the same shape.
        temperature: Strictly positive finite softmax temperature.

    Returns:
        Square EEG-query by movement-candidate logit matrix.
    """
    if eeg.ndim != 2 or eeg.shape != movement.shape or len(eeg) < 2 or eeg.shape[1] < 1:
        raise ValueError("Expected matching (N, D) embeddings with N >= 2 and D >= 1")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if not torch.isfinite(eeg).all() or not torch.isfinite(movement).all():
        raise ValueError("Nonfinite embeddings")
    return functional.normalize(eeg, dim=-1) @ functional.normalize(movement, dim=-1).T / temperature


def positive_columns(scores: torch.Tensor, positives: torch.Tensor | None = None) -> torch.Tensor:
    """Validate one positive column per query, with the diagonal as the default."""
    if scores.ndim != 2 or min(scores.shape) < 1 or not torch.isfinite(scores).all():
        raise ValueError("Expected a finite, nonempty query-by-candidate matrix")
    if positives is None:
        if scores.shape[0] != scores.shape[1]:
            raise ValueError("Non-square galleries require explicit positive columns")
        positives = torch.arange(len(scores), device=scores.device)
    if positives.shape != (len(scores),) or positives.dtype != torch.long or positives.device != scores.device:
        raise ValueError("Positive columns must be device-matched int64 indices, one per query")
    if bool(((positives < 0) | (positives >= scores.shape[1])).any()):
        raise ValueError("Positive column outside gallery")
    return positives


def symmetric_loss(logits: torch.Tensor, positives: torch.Tensor | None = None) -> torch.Tensor:
    """Average EEG-to-movement and movement-to-EEG hard-target cross entropy.

    Positives must be a permutation: every candidate is paired with exactly one query.
    Temporal or behaviour-similarity soft targets are a future, separately tested objective.
    """
    positives = positive_columns(logits, positives)
    if logits.shape[0] != logits.shape[1] or len(torch.unique(positives)) != len(positives):
        raise ValueError("Symmetric loss requires a square one-to-one pairing")
    if len(positives) < 2:
        raise ValueError("Contrastive loss requires at least two trials")
    reverse = torch.argsort(positives)
    return (functional.cross_entropy(logits, positives) + functional.cross_entropy(logits.T, reverse)) / 2


@torch.no_grad()
def retrieval_metrics(scores: torch.Tensor, positives: torch.Tensor | None = None) -> dict[str, float]:
    """Return top-1 and MRR under uniform random ordering within exact score ties.

    This averages reciprocal ranks over all ranks occupied by a tied block, rather than
    taking the reciprocal of its mean rank. Rows are queries; columns are candidates.
    """
    positives = positive_columns(scores, positives)
    value = scores.gather(1, positives[:, None])
    better = (scores > value).sum(1)
    tied = (scores == value).sum(1)
    ranks = torch.arange(1, scores.shape[1] + 1, dtype=torch.float64, device=scores.device)
    harmonic = torch.cat([torch.zeros(1, dtype=torch.float64, device=scores.device), (1 / ranks).cumsum(0)])
    reciprocal = (harmonic[better + tied] - harmonic[better]) / tied
    top1 = torch.where(better == 0, 1 / tied.double(), 0.0)
    return dict(
        top1=float(top1.mean()),
        mrr=float(reciprocal.mean()),
        chance_top1=1 / scores.shape[1],
        chance_mrr=float(harmonic[-1] / scores.shape[1]),
    )
