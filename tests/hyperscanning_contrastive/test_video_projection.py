"""Centering leakage, head structure and matched video-head experiment controls."""

import json

import numpy as np
import torch
from test_video_labram import reference  # noqa: F401

from hyperscanning_contrastive.video_projection import VideoHead, center_bank, run_video_projection


def test_centering():
    """Use matched training content for subject calibration, never validation samples."""
    train = {
        (p, t): np.full((4, 200), offset + t, dtype=np.float32)
        for p, offset in [("a", 10), ("b", 20)]
        for t in range(5)
    }
    train[("a", 9)] = np.full((4, 200), 999, dtype=np.float32)
    bank = dict(train=train, validation={("a", 12): np.full((4, 200), 1000, dtype=np.float32)})
    centered, centers, common = center_bank(bank, "participant")
    assert common == list(range(5))
    np.testing.assert_allclose(centers["a"], 12)
    np.testing.assert_allclose(centers["b"], 22)
    np.testing.assert_allclose(centered["validation"][("a", 12)], 988)
    _, global_centers, _ = center_bank(bank, "global")
    bank["validation"][("a", 12)] *= -999
    _, changed, _ = center_bank(bank, "global")
    np.testing.assert_array_equal(global_centers["global"], changed["global"])


def test_heads():
    """Both structured and pooled heads expose finite learnable gradients."""
    for profile, shape in [("pooled", (8, 200)), ("temporal", (8, 4, 200))]:
        model = VideoHead(profile)
        x = torch.randn(shape, requires_grad=True)
        y = model(x)
        assert y.shape == (8, 64)
        y.square().mean().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert sum(p.numel() for p in model.parameters()) < 20000


def test_pipeline(reference, tmp_path, monkeypatch):
    """Run real heads with synthetic frozen features and unchanged reserved-person exclusion."""
    cache, run = reference
    metadata = json.loads((run / "baseline.json").read_text())
    metadata["metrics"] = {}
    (run / "baseline.json").write_text(json.dumps(metadata))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"test")

    def extract(raw, checkpoint, device, batch_size, output):
        """Supply deterministic token features only for eligible development windows."""
        assert all(p != "reserved" for values in raw.values() for p, _ in values)
        rng = np.random.default_rng(2)
        banks = {profile: {s: {} for s in raw} for profile in ("pooled", "temporal")}
        for split, values in raw.items():
            for key in values:
                z = rng.normal(size=(4, 200)).astype(np.float32)
                banks["temporal"][split][key] = z
                banks["pooled"][split][key] = z.mean(0)
        return banks, 0.0

    monkeypatch.setattr("hyperscanning_contrastive.video_projection.extract_banks", extract)
    result = run_video_projection(cache, run, tmp_path / "result", checkpoint, epochs=1, steps_per_epoch=2, pairs=4)
    assert len(result["experiments"]) == 4
    assert result["training_person_windows"] == 8
    for experiment in result["experiments"].values():
        assert experiment["arms"]["true"]["selected"]["epoch"] in (0, 1)
    assert (tmp_path / "result" / "comparison.png").is_file()


def test_raw_inputs_preserved(reference):
    """LaBraM receives original microvolts without CNN demeaning or channel scaling."""
    from hyperscanning_contrastive.video_cnn import load_bank
    from hyperscanning_contrastive.video_labram import load_reference

    cache, run = reference
    _, _, windows = load_reference(cache, run)
    bank, scale = load_bank(cache, run, windows, standardize=False)
    original = np.load(cache / "a.npy")[:, 800:1600]
    np.testing.assert_array_equal(bank["train"][("a", 800)], original)
    np.testing.assert_array_equal(scale, np.ones(64))
