"""Symmetric contrastive loss and retrieval metrics for explicitly aligned pairs."""

import torch
from torch import Tensor
from torch.nn import functional


def pair_scores(first: Tensor, second: Tensor) -> Tensor:
    """Compute cosine scores for two equally sized matrices of paired embeddings."""
    if first.ndim != 2 or first.shape != second.shape or len(first) < 2:
        raise ValueError("Expect matching embedding matrices with at least two pairs.")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("Embeddings must be finite.")
    return functional.normalize(first, dim=1) @ functional.normalize(second, dim=1).T


def masked_scores(first: Tensor, second: Tensor, allowed: Tensor | None = None) -> Tensor:
    """Mask ineligible candidates while preserving diagonal positives and negatives in both directions."""
    scores = pair_scores(first, second)
    if allowed is None:
        return scores
    if allowed.shape != scores.shape or allowed.dtype != torch.bool:
        raise ValueError("Candidate mask must be a boolean square matrix matching the batch.")
    if not allowed.diagonal().all() or (allowed.sum(0) < 2).any() or (allowed.sum(1) < 2).any():
        raise ValueError("Every anchor must retain its positive and at least one negative in both directions.")
    return scores.masked_fill(~allowed, -torch.inf)


def contrastive_loss(first: Tensor, second: Tensor, temperature: float = 0.1, allowed: Tensor | None = None) -> Tensor:
    """Compute symmetric InfoNCE with diagonal positives and an explicit eligible-candidate mask."""
    if not 0 < temperature < float("inf"):
        raise ValueError("Temperature must be finite and positive.")
    logits = masked_scores(first, second, allowed) / temperature
    targets = torch.arange(len(first), device=first.device)
    return (functional.cross_entropy(logits, targets) + functional.cross_entropy(logits.T, targets)) / 2


def retrieval(first: Tensor, second: Tensor, allowed: Tensor | None = None) -> dict[str, float]:
    """Report bidirectional retrieval using average ranks and fractional top-one credit for ties."""
    scores = masked_scores(first, second, allowed)
    positives = scores.diagonal()
    ranks, top1 = [], []
    for matrix in (scores, scores.T):
        greater = (matrix > positives[:, None]).sum(1)
        equal = (matrix == positives[:, None]).sum(1)
        ranks.append(1 + greater + (equal - 1) / 2)
        top1.append((greater == 0).float() / equal)
    return {
        "top1": torch.cat(top1).mean().item(),
        "mean_rank": torch.cat(ranks).float().mean().item(),
    }
