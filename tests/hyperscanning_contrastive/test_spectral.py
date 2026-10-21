"""Spectral feature correctness and matched trial-retrieval controls."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from hyperscanning_contrastive.adaptation import train_adaptation
from hyperscanning_contrastive.spectral import FeatureStandardizer, bandpower_features, train_spectral
from hyperscanning_contrastive.ssl import EEGCache
from tests.hyperscanning_contrastive.test_adaptation import TinyAdaptiveEncoder
from tests.hyperscanning_contrastive.test_ssl import make_cache


def test_spectral_power_and_scaling() -> None:
    """A 10 Hz tone peaks in alpha; amplitude scaling gives the expected log-power change."""
    signal = np.broadcast_to(np.sin(2 * np.pi * 10 * np.arange(800) / 200), (2, 64, 800)).copy()
    features = bandpower_features(signal).reshape(2, 64, 5)
    assert np.all(features.argmax(-1) == 2)
    scaled = bandpower_features(2 * signal).reshape(2, 64, 5)
    np.testing.assert_allclose(scaled[..., 2] - features[..., 2], np.log10(4), atol=1e-6)
    assert np.isfinite(bandpower_features(np.zeros_like(signal))).all()


def test_standardization_training_only() -> None:
    """Fit one fixed transform jointly on both participants and reject validation banks."""
    bank = {
        0: {"split": "train", "a": torch.tensor([[1.0, 2.0], [3.0, 2.0]]), "b": torch.tensor([[5.0, 2.0], [7.0, 2.0]])}
    }
    scaler = FeatureStandardizer(bank)
    torch.testing.assert_close(scaler.mean, torch.tensor([4.0, 2.0]))
    before = {name: value.clone() for name, value in scaler.state_dict().items()}
    assert torch.isfinite(scaler(torch.ones(2, 2) * 100)).all()
    for name, value in scaler.state_dict().items():
        torch.testing.assert_close(value, before[name])
    bank[0]["split"] = "validation"
    with pytest.raises(ValueError, match="training features only"):
        FeatureStandardizer(bank)


def test_spectral_reference_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the exact LaBraM galleries and fit only training spectra; exclude held-out dyads."""
    cache = make_cache(tmp_path / "cache")
    reference = tmp_path / "reference"
    monkeypatch.setattr("hyperscanning_contrastive.adaptation.ContrastiveLabram", TinyAdaptiveEncoder)
    previous = train_adaptation(
        cache,
        reference,
        tmp_path / "absent.pth",
        initialization="random",
        pairs=2,
        epochs=1,
        steps_per_epoch=2,
        validation_protocol="within-dyad",
        validation_trials=2,
    )
    original = EEGCache.signals

    def checked_signals(self: EEGCache, frame: pd.DataFrame) -> torch.Tensor:
        """Forbid any spectral extraction from external validation or test dyads."""
        assert frame.split.isin(["train", "validation"]).all()
        return original(self, frame)

    monkeypatch.setattr(EEGCache, "signals", checked_signals)
    output = tmp_path / "spectral"
    result = train_spectral(cache, output, reference)
    for key in ("gallery_sha256", "trial_splits_sha256", "shuffled_training_pairs_sha256"):
        assert result[key] == previous[key]
    assert not result["test_evaluated"]
    initial = torch.load(output / "initial_projection.pt", weights_only=True)
    for arm in ("true", "shuffled"):
        history = pd.read_csv(output / arm / "history.csv")
        assert result["arms"][arm]["best_epoch"] == history.mrr.idxmax()
        last = torch.load(output / arm / "last_projection.pt", weights_only=True)
        torch.testing.assert_close(last["0.mean"], initial["0.mean"])
        torch.testing.assert_close(last["0.scale"], initial["0.scale"])
        assert not torch.equal(last["1.weight"], initial["1.weight"])
    assert (
        pd.read_csv(output / "true/history.csv").iloc[0].mrr == pd.read_csv(output / "shuffled/history.csv").iloc[0].mrr
    )
    (reference / "galleries.csv").write_text("tampered\n")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        train_spectral(cache, tmp_path / "invalid", reference)
