"""CNN sampling and train-only normalization regression checks."""

import json

import numpy as np
import torch
from test_video_labram import reference  # noqa: F401

from hyperscanning_contrastive.video_cnn import VideoCNN, batch, make_schedule, run_video_cnn


def test_schedule():
    """Schedules contain distinct moments and shuffled inputs are exactly reordered originals."""
    bank = {(p, t): np.full((64, 800), t, dtype=np.float32) for p in ("a", "b") for t in range(9)}
    schedule, census = make_schedule(bank, 8, 10, 42)
    assert schedule == make_schedule(bank, 8, 10, 42)[0]
    assert census.eligible.sum() == 1
    for entry in schedule:
        x, y = batch(bank, entry, "cpu")
        xx, yy = batch(bank, entry, "cpu", True)
        assert torch.equal(x, xx)
        assert not torch.any(y[:, 0, 0] == yy[:, 0, 0])
        assert torch.equal(y[:, 0, 0].sort().values, yy[:, 0, 0].sort().values)
        assert len(set(entry[2])) == 8


def test_model():
    """The encoder preserves a finite gradient path from raw temporal signals."""
    model = VideoCNN()
    x = torch.randn(4, 64, 800, requires_grad=True)
    z = model(x)
    assert z.shape == (4, 64)
    z.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_end_to_end(reference, tmp_path):
    """Fit a real small CNN through both arms without ever opening reserved signals."""
    cache, run = reference
    metadata = json.loads((run / "baseline.json").read_text())
    metadata["metrics"] = {}
    (run / "baseline.json").write_text(json.dumps(metadata))
    report = run_video_cnn(cache, run, tmp_path / "cnn", epochs=1, steps_per_epoch=2, pairs=4, tiny_steps=200)
    assert report["training_person_windows"] == 8
    assert not report["reserved_people_evaluated"]
    assert set(report["arms"]) == {"true", "misaligned"}
    assert (tmp_path / "cnn" / "learning_curves.png").is_file()
