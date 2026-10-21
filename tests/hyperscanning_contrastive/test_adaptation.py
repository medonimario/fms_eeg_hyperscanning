"""Partial encoder training, paired-control reproducibility, and test isolation."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from hyperscanning_contrastive.adaptation import configure_adaptation, train_adaptation
from hyperscanning_contrastive.alignment import train_alignment
from hyperscanning_contrastive.ssl import EEGCache
from tests.hyperscanning_contrastive.test_ssl import make_cache


class TinyAdaptiveEncoder(nn.Module):
    """Small transformer-shaped stand-in for orchestration tests."""

    def __init__(self, initialization: str, checkpoint: Path, train_backbone: bool) -> None:
        """Build a frozen input layer followed by three trainable block candidates."""
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.input = nn.Linear(64, 8)
        self.backbone.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.backbone.fc_norm = nn.LayerNorm(8)
        self.backbone.requires_grad_(train_backbone)
        self.projection = nn.Linear(8, 8)

    def features(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute nonlinear features that exercise gradients through all blocks."""
        values = self.backbone.input(signal.mean(-1))
        for block in self.backbone.blocks:
            values = torch.tanh(block(values))
        return self.backbone.fc_norm(values)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Project the current encoder output."""
        return self.projection(self.features(signal))


def test_adaptation_matches_frozen_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep both control starts/sampling contracts identical and never forward test signals."""
    cache = make_cache(tmp_path / "cache")
    monkeypatch.setattr("hyperscanning_contrastive.adaptation.ContrastiveLabram", TinyAdaptiveEncoder)
    monkeypatch.setattr("hyperscanning_contrastive.alignment.ContrastiveLabram", TinyAdaptiveEncoder)
    original_signals = EEGCache.signals

    def checked_signals(self: EEGCache, frame: pd.DataFrame) -> torch.Tensor:
        """Fail if evaluation or fitting reads test EEG."""
        assert not frame.split.eq("test").any()
        return original_signals(self, frame)

    monkeypatch.setattr(EEGCache, "signals", checked_signals)
    adapted, frozen = tmp_path / "adapted", tmp_path / "frozen"
    options = dict(initialization="random", pairs=2, epochs=2, steps_per_epoch=3)
    report = train_adaptation(cache, adapted, tmp_path / "absent.pth", **options)
    reference = train_alignment(cache, frozen, tmp_path / "absent.pth", **options)
    assert report["gallery_sha256"] == reference["gallery_sha256"]
    assert report["shuffled_training_pairs_sha256"] == reference["shuffled_training_pairs_sha256"]
    assert report["initial_projection"] == reference["initial_projection"]
    assert report["checkpoint_sha256"] is None
    assert not report["test_evaluated"]
    initial = torch.load(adapted / "initial.pt", weights_only=True)
    for arm in ("true", "shuffled"):
        history = pd.read_csv(adapted / arm / "history.csv")
        assert report["arms"][arm]["best_epoch"] == history.mrr.idxmax()
        final = torch.load(adapted / arm / "last.pt", weights_only=True)
        for name in initial:
            if name not in report["trainable_parameters"]:
                assert torch.equal(initial[name], final[name]), name
        for prefix in ("backbone.blocks.1", "backbone.blocks.2", "backbone.fc_norm", "projection"):
            assert any(not torch.equal(initial[name], final[name]) for name in initial if name.startswith(prefix))
    true = pd.read_csv(adapted / "true/history.csv").iloc[0]
    shuffled = pd.read_csv(adapted / "shuffled/history.csv").iloc[0]
    for name in ("mrr", "validation_loss", "train_true_loss"):
        assert true[name] == shuffled[name]


def test_invalid_unfreeze_count(tmp_path: Path) -> None:
    """Reject unsupported block counts rather than silently unfreezing all blocks."""
    model = TinyAdaptiveEncoder("random", tmp_path / "absent.pth", False)
    for blocks in (0, -1, 4):
        with pytest.raises(ValueError, match="Unfreeze count"):
            configure_adaptation(model, blocks)
