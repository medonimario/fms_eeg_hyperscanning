"""Independent mathematical checks for the new project's contrastive primitives."""

import math
import subprocess
import sys

import pytest
import torch
from torch.nn import functional

from cl_behaviour.contrastive import paired_logits, retrieval_metrics, symmetric_loss
from cl_behaviour.controls import fit_arm, independent_pairing_null, synthetic_pairs
from cl_behaviour.models import LinearHeads


def test_pairing_and_candidate_order_invariance():
    """Permuting candidate columns and positive indices leaves loss and metrics unchanged."""
    eeg = torch.eye(4, requires_grad=True)
    movement = torch.eye(4, requires_grad=True)
    matrix = paired_logits(eeg, movement)
    expected = functional.cross_entropy(matrix, torch.arange(4))
    torch.testing.assert_close(symmetric_loss(matrix), expected)
    order = torch.tensor([2, 0, 3, 1])
    positive = torch.argsort(order)
    torch.testing.assert_close(symmetric_loss(matrix[:, order], positive), expected)
    assert retrieval_metrics(matrix[:, order], positive) == retrieval_metrics(matrix)
    loss = symmetric_loss(matrix)
    loss.backward()
    assert eeg.grad is not None and movement.grad is not None
    assert torch.isfinite(eeg.grad).all() and torch.isfinite(movement.grad).all()
    assert eeg.grad.norm() > 0 and movement.grad.norm() > 0


def test_exact_tie_averaging_and_direction():
    """All-equal scores reproduce chance, and partial ties average reciprocal ranks."""
    for size in (3, 4, 8):
        values = retrieval_metrics(torch.zeros(size, size))
        assert values["top1"] == pytest.approx(1 / size)
        assert values["mrr"] == pytest.approx(sum(1 / rank for rank in range(1, size + 1)) / size)
    values = retrieval_metrics(torch.tensor([[4.0, 3.0, 3.0, 0.0]]), torch.tensor([1]))
    assert values["top1"] == 0
    assert values["mrr"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    scores = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    assert retrieval_metrics(scores)["top1"] == 0.5
    assert retrieval_metrics(scores.T)["top1"] == 0.5


def test_invalid_pairings_and_temperature():
    """Reject malformed supervision rather than silently changing the objective."""
    for temperature in (0, -1, math.nan):
        with pytest.raises(ValueError, match="Temperature"):
            paired_logits(torch.eye(2), torch.eye(2), temperature)
    with pytest.raises(ValueError, match="one-to-one"):
        symmetric_loss(torch.eye(3), torch.tensor([0, 0, 2]))
    with pytest.raises(ValueError, match="Nonfinite"):
        paired_logits(torch.full((2, 2), math.nan), torch.eye(2))


def test_synthetic_generalization_and_independent_null():
    """A shared latent relationship is learnable on unseen trials; randomized labels give chance."""
    data = synthetic_pairs(42)
    assert data[0].mean(0).abs().max() < 1e-5
    assert data[1].mean(0).abs().max() < 1e-5
    with torch.random.fork_rng():
        torch.manual_seed(1042)
        initial = LinearHeads(48, 32)
    metrics, _, matrices = fit_arm(initial, data, torch.arange(256), 100)
    assert metrics["eeg_to_movement_top1"] >= 0.9
    assert metrics["movement_to_eeg_top1"] >= 0.9
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert independent_pairing_null(matrices, 3042, repeats=128)["passed"]


def test_controls_import_without_other_projects():
    """Block cross-project imports, including indirect imports through the CLI."""
    script = """
import sys
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'hyperscanning_contrastive', 'labram_classification', 'mTRF', 'mne'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import cl_behaviour.controls
import cl_behaviour.tiny_fit
import cl_behaviour.training
import cl_behaviour.__main__
sys.argv = ['cl_behaviour', 'controls', '--help']
cl_behaviour.__main__.main()
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
