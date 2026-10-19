"""LaBraM feature-adapter checks using the installed native architecture."""

from pathlib import Path

import torch

from hyperscanning_contrastive.data import CHANNELS
from hyperscanning_contrastive.overfit import ContrastiveLabram


def test_random_labram_features_and_backward(tmp_path: Path) -> None:
    """Exercise real channel-aware LaBraM with nonexistent checkpoint and encoder gradients."""
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = ContrastiveLabram("random", tmp_path / "does-not-exist.pth", train_backbone=True)
        model.eval()
        signal = torch.randn(2, 64, 800) * 10
        with torch.no_grad():
            expected = model.backbone(signal / 100, ch_names=CHANNELS)
            actual = model.features(signal)
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (2, 200)
        output = model(signal)
        assert output.shape == (2, 128)
        output.square().mean().backward()
        gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    finally:
        torch.set_num_threads(threads)
