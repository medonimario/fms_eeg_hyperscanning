"""Verify reproducible scratch LaBraM construction without accessing pretrained weights."""

from pathlib import Path

import pytest
import torch

from labram_classification import model
from labram_classification.__main__ import parser
from labram_classification.training import BinaryLoss, resolve_model_options


def test_random_initialization_without_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Random initialization is seeded, nonconstant, trainable, and never loads student weights."""

    def forbidden_read(*args: object, **kwargs: object) -> None:
        """Fail on any attempt to read a pretrained checkpoint."""
        pytest.fail("Random initialization must not read pretrained weights")

    monkeypatch.setattr(model, "read_student_weights", forbidden_read)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    first = model.load_model(tmp_path / "absent.pth", 0, initialization="random")
    reference = first.backbone.blocks[0].attn.qkv.weight.detach().clone()
    torch.manual_seed(7)
    second = model.load_model(tmp_path / "absent.pth", 0, initialization="random")
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    del second
    torch.manual_seed(8)
    second = model.load_model(tmp_path / "absent.pth", 0, initialization="random")
    assert not torch.equal(reference, second.backbone.blocks[0].attn.qkv.weight)
    del second
    logits = first(torch.randn(2, 64, 800))
    BinaryLoss()(logits, torch.tensor([0, 1])).backward()
    assert logits.shape == (2,)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in first.parameters())


def test_initialization_profiles() -> None:
    """Keep old defaults, expose scratch defaults and allow matched-protocol overrides."""
    base = ["cv", "--cache", "cache", "--output", "out"]
    pretrained = resolve_model_options(vars(parser().parse_args(base)))
    assert pretrained["initialization"] == "pretrained"
    assert (pretrained["epochs"], pretrained["layer_decay"]) == (10, 0.65)
    scratch = resolve_model_options(vars(parser().parse_args(base + ["--initialization", "random"])))
    assert (scratch["epochs"], scratch["lr"], scratch["layer_decay"]) == (100, 1e-4, 1)
    matched = resolve_model_options(dict(model="labram", initialization="random", epochs=10, layer_decay=0.65))
    assert (matched["epochs"], matched["layer_decay"]) == (10, 0.65)
    with pytest.raises(ValueError, match="EEGNet supports only"):
        resolve_model_options(dict(model="eegnet", initialization="pretrained"))
