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


def test_native_partial_adaptation(tmp_path: Path) -> None:
    """Verify real LaBraM partial freezing and nonzero gradients in the selected blocks."""
    from hyperscanning_contrastive.adaptation import configure_adaptation
    from hyperscanning_contrastive.contrastive import contrastive_loss

    model = ContrastiveLabram("random", tmp_path / "absent.pth", train_backbone=False).eval()
    names = configure_adaptation(model, 2)
    last = len(model.backbone.blocks) - 1
    expected = (f"backbone.blocks.{last - 1}.", f"backbone.blocks.{last}.", "backbone.fc_norm.", "projection.")
    assert names and all(name.startswith(expected) for name in names)
    a, b = model(torch.randn(2, 64, 800)), model(torch.randn(2, 64, 800))
    contrastive_loss(a, b).backward()
    for prefix in expected:
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        assert any(g is not None and g.abs().sum() > 0 for g in gradients)
        assert all(g is None or torch.isfinite(g).all() for g in gradients)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
