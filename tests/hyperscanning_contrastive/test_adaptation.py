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


def test_within_dyad_trial_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Train only on assigned trials and never forward external validation or test dyads."""
    from hyperscanning_contrastive.ssl import SSLConfig
    from hyperscanning_contrastive.trial_split import split_within_dyad

    path = make_cache(tmp_path / "cache")
    config = SSLConfig(initialization="random", pairs=2)
    first, second = EEGCache(path, config), EEGCache(path, config)
    split_within_dyad(first, 2, 2, 2042)
    split_within_dyad(second, 2, 2, 2042)
    pd.testing.assert_frame_equal(first.rows, second.rows)
    train = first.rows[first.rows.split == "train"]
    validation = first.rows[first.rows.split == "validation"]
    assert set(train.group) == set(validation.group) == {0}
    assert set(train.trial_id).isdisjoint(validation.trial_id)
    assert validation.groupby(["group", "condition"]).size().eq(2).all()
    for split in ("train", "validation"):
        frames = first.strata[split]
        for a, b in zip(frames[::2], frames[1::2]):
            assert a.trial_id.tolist() == b.trial_id.tolist()
    original_signals = EEGCache.signals
    forwarded = {"train": set(), "validation": set()}

    def checked_signals(self: EEGCache, frame: pd.DataFrame) -> torch.Tensor:
        """Require held-out trial exclusion whenever autograd is enabled."""
        split = frame.split.iloc[0]
        assert split in {"train", "validation"}
        if torch.is_grad_enabled():
            assert split == "train"
        forwarded[split].update(frame.trial_id)
        return original_signals(self, frame)

    monkeypatch.setattr(EEGCache, "signals", checked_signals)
    monkeypatch.setattr("hyperscanning_contrastive.adaptation.ContrastiveLabram", TinyAdaptiveEncoder)
    output = tmp_path / "run"
    report = train_adaptation(
        path,
        output,
        tmp_path / "absent.pth",
        initialization="random",
        pairs=2,
        epochs=2,
        steps_per_epoch=3,
        validation_protocol="within-dyad",
        validation_trials=2,
    )
    assert forwarded["train"].isdisjoint(forwarded["validation"])
    assert forwarded["validation"] == set(validation.trial_id)
    assert report["validation_fold"] is None
    assert report["reserved_external_validation_fold"] == 1
    assert report["split_counts"]["validation"] == {"dyads": 1, "paired_windows": 6}
    assert not report["test_evaluated"] and not report["external_validation_evaluated"]
    manifest = pd.read_csv(output / "trial_splits.csv")
    pd.testing.assert_frame_equal(manifest, first.rows[manifest.columns], check_index_type=False)
    mapping = pd.read_csv(output / "shuffled_training_pairs.csv")
    assert set(mapping.a_trial_id) == set(mapping.b_trial_id) == set(train.trial_id)
    galleries = pd.read_csv(output / "galleries.csv")
    assert set(galleries.trial_id) == set(validation.trial_id)
    for arm in ("true", "shuffled"):
        queries = pd.read_csv(output / arm / "validation_best.csv")
        assert set(queries.trial_id) == set(validation.trial_id)
        history = pd.read_csv(output / arm / "history.csv")
        assert report["arms"][arm]["best_epoch"] == history.mrr.idxmax()
    true = pd.read_csv(output / "true/history.csv").iloc[0]
    shuffled = pd.read_csv(output / "shuffled/history.csv").iloc[0]
    assert true.mrr == shuffled.mrr


def test_within_dyad_insufficient_trials(tmp_path: Path) -> None:
    """Reject short strata without silently shrinking galleries or dropping dyads."""
    from hyperscanning_contrastive.ssl import SSLConfig
    from hyperscanning_contrastive.trial_split import split_within_dyad

    cache = EEGCache(make_cache(tmp_path / "cache"), SSLConfig(pairs=2))
    original = cache.rows.copy()
    with pytest.raises(ValueError, match="cannot supply"):
        split_within_dyad(cache, 3, 2, 2042)
    pd.testing.assert_frame_equal(cache.rows, original)
    with pytest.raises(ValueError, match="gallery size"):
        split_within_dyad(cache, 1, 2, 2042)
