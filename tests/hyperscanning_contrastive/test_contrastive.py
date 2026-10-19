"""Engineering tests for contrastive loss, masking, ranking, and synthetic learning."""

from pathlib import Path

import pytest
import torch

from hyperscanning_contrastive.contrastive import contrastive_loss, retrieval
from hyperscanning_contrastive.diagnostics import run_controls


def test_ties_do_not_reward_collapse() -> None:
    """Collapsed embeddings receive chance credit rather than perfect diagonal retrieval."""
    values = torch.ones(8, 4)
    assert retrieval(values, values) == {"top1": 0.125, "mean_rank": 4.5}
    assert contrastive_loss(values, values).item() == pytest.approx(float(torch.log(torch.tensor(8.0))))


def test_mask_and_gradients() -> None:
    """Exclude non-candidates in both directions and backpropagate finite gradients."""
    values = torch.eye(4, requires_grad=True)
    allowed = torch.tensor([[True, True, False, False]] * 2 + [[False, False, True, True]] * 2)
    loss = contrastive_loss(values, values, allowed=allowed)
    expected = torch.logsumexp(torch.tensor([10.0, 0.0]), 0) - 10
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    loss.backward()
    assert torch.isfinite(values.grad).all()
    assert retrieval(values, values, allowed)["top1"] == 1
    with pytest.raises(ValueError, match="negative"):
        contrastive_loss(values, values, allowed=torch.eye(4, dtype=torch.bool))


def test_bad_positive_mask() -> None:
    """Reject a mask that hides the designated positive."""
    with pytest.raises(ValueError, match="positive"):
        contrastive_loss(torch.eye(4), torch.eye(4), allowed=~torch.eye(4, dtype=torch.bool))


def test_synthetic_controls(tmp_path: Path) -> None:
    """Learn shared latent structure on disjoint training/test examples and calibrate independence."""
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report = run_controls(tmp_path / "controls")
    finally:
        torch.set_num_threads(threads)
    assert report["passed"], report
    assert (tmp_path / "controls" / "controls.json").exists()
