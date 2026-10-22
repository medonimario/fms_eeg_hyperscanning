"""Patch order, pooling equivalence, split isolation, and matched readout checks."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from test_partner_windows import window_cache
from torch import nn

from hyperscanning_contrastive import partner_patches as pp
from hyperscanning_contrastive.data import CHANNELS
from hyperscanning_contrastive.overfit import ContrastiveLabram
from hyperscanning_contrastive.ssl import EEGCache


def test_real_labram_tokens() -> None:
    """Actual Braindecode tokens reconstruct pooling and preserve channel-major segmentation."""
    from braindecode.models import Labram

    model = ContrastiveLabram.__new__(ContrastiveLabram)
    nn.Module.__init__(model)
    model.backbone = Labram(n_times=800, n_chans=128, n_outputs=0, num_layers=1, use_mean_pooling=True)
    model.backbone.final_layer = nn.Identity()
    model.requires_grad_(False).eval()
    extractor = pp.TokenExtractor(model)
    x = torch.randn(2, 64, 800)
    with torch.no_grad():
        tokens = extractor.features(x)
        pooled = pp.reduce_tokens(tokens, "pooled", model.backbone.fc_norm)
        torch.testing.assert_close(pooled, model.features(x))
        patches = model.backbone.patch_embed.segment_patch(x)
    assert patches.shape == (2, 64, 4, 200)
    torch.testing.assert_close(patches[:, 17, 2], x[:, 17, 400:600])
    assert len(CHANNELS) == 64
    assert tokens.shape == (2, 64, 4, 200)
    assert extractor.max_reconstruction_error < 1e-6


def test_reductions() -> None:
    """Only the intended axes disappear; full features retain channel and time order."""
    values = torch.arange(64 * 4 * 200, dtype=torch.float32).reshape(1, 64, 4, 200)
    norm = nn.Identity()
    torch.testing.assert_close(pp.reduce_tokens(values, "spatial", norm), values.mean(2))
    torch.testing.assert_close(pp.reduce_tokens(values, "temporal", norm), values.mean(1))
    full = pp.reduce_tokens(values, "full", norm)
    torch.testing.assert_close(full[:, 17 * 4 + 2], values[:, 17, 2])


class FakeBackbone(nn.Module):
    """Small deterministic contextual-token stand-in for end-to-end readout tests."""

    def __init__(self) -> None:
        """Provide the two normalization stages of mean-pooling LaBraM."""
        super().__init__()
        self.norm = nn.Identity()
        self.fc_norm = nn.LayerNorm(200)


class FakeEncoder(nn.Module):
    """Expose the production feature contract without loading checkpoint weights."""

    def __init__(self, *args: object) -> None:
        """Freeze the fixture backbone."""
        super().__init__()
        self.backbone = FakeBackbone().requires_grad_(False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Emit ordered token values derived from the input EEG."""
        tokens = x.reshape(-1, 256, 200)
        cls = tokens.mean(1, keepdim=True)
        values = self.backbone.norm(torch.cat((cls, tokens), dim=1))
        return self.backbone.fc_norm(values[:, 1:].mean(1))


def test_patch_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Extract eligible EEG once, fit all heads, and retain identical control manifests."""
    path = window_cache(tmp_path / "cache")
    monkeypatch.setattr(pp, "ContrastiveLabram", FakeEncoder)
    original = EEGCache.signals
    seen = []

    def signals(self: EEGCache, rows: pd.DataFrame) -> torch.Tensor:
        """Disallow closed and reserved EEG during extraction."""
        assert set(rows.condition) == {"open"}
        assert set(rows.fold) == {0}
        seen.extend((r.trial_id, r.crop_start_s, r.member) for r in rows.itertuples())
        return original(self, rows)

    monkeypatch.setattr(EEGCache, "signals", signals)
    output = tmp_path / "run"
    report = pp.run_partner_patches(
        path, output, tmp_path / "absent", initialization="random", validation_trials=2, epochs=1
    )
    assert len(seen) == len(set(seen))
    expected = {"pooled": 12800, "spatial": 68736, "temporal": 7296, "full": 265344}
    for profile, item in report["profiles"].items():
        assert item["trainable_parameters"] == expected[profile]
        assert item["manifest_sha256"] == report["profiles"]["pooled"]["manifest_sha256"]
        first = torch.load(output / profile / "initial_projection.pt", weights_only=True)
        for arm in ("true", "shuffled"):
            last = torch.load(output / profile / arm / "last_projection.pt", weights_only=True)
            torch.testing.assert_close(first["0.mean"], last["0.mean"])
            assert not torch.equal(first["1.weight"], last["1.weight"])
    assert (output / "patch_comparison.png").stat().st_size > 1000


def test_tuning_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nondefault CLI values reach optimizer and every scoring/training loss for a single profile."""
    import sys

    from hyperscanning_contrastive import __main__ as cli
    from hyperscanning_contrastive import partner_windows as pw

    path = window_cache(tmp_path / "cache")
    output = tmp_path / "run"
    monkeypatch.setattr(pp, "ContrastiveLabram", FakeEncoder)
    optimizer = torch.optim.AdamW
    loss = pw.contrastive_loss
    score = pw.score_windows
    observed = {"optimizer": 0, "loss": 0, "score": 0}

    def checked_optimizer(params, **kwargs):
        """Assert both arms use the requested optimization settings."""
        assert kwargs == {"lr": 0.0001, "weight_decay": 0.0}
        observed["optimizer"] += 1
        return optimizer(params, **kwargs)

    def checked_loss(a, b, temperature):
        """Assert training receives the requested temperature."""
        assert temperature == 0.2
        observed["loss"] += 1
        return loss(a, b, temperature)

    def checked_score(head, bank, temperature, mappings=None):
        """Assert baselines and diagnostics use the same temperature."""
        assert temperature == 0.2
        observed["score"] += 1
        return score(head, bank, temperature, mappings)

    monkeypatch.setattr(torch.optim, "AdamW", checked_optimizer)
    monkeypatch.setattr(pw, "contrastive_loss", checked_loss)
    monkeypatch.setattr(pw, "score_windows", checked_score)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hyperscanning_contrastive",
            "partner-patches",
            "--cache",
            str(path),
            "--output",
            str(output),
            "--initialization",
            "random",
            "--validation-trials",
            "2",
            "--epochs",
            "1",
            "--temperature",
            "0.2",
            "--projection-lr",
            "0.0001",
            "--weight-decay",
            "0",
            "--readout-profiles",
            "full",
        ],
    )
    cli.main()
    assert observed["optimizer"] == 2
    assert observed["loss"] > 0 and observed["score"] > 0
    assert (output / "full" / "partner_windows.json").exists()
    assert not (output / "pooled").exists()
    assert (output / "patch_comparison.png").exists()


@pytest.mark.parametrize(
    "values",
    [(0, 0.001, 0.01), (0.05, -1, 0.01), (0.05, 0.001, -1), (float("nan"), 0.001, 0.01), (0.05, float("inf"), 0.01)],
)
def test_invalid_tuning(values: tuple[float, float, float]) -> None:
    """Fail before loading any model or cache on invalid optimization settings."""
    from hyperscanning_contrastive.partner_windows import validate_optimization

    with pytest.raises(ValueError, match="Invalid"):
        validate_optimization(*values)
