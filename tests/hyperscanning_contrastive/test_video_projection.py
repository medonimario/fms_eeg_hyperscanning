"""Centering leakage, head structure and matched video-head experiment controls."""

import json

import numpy as np
import pytest
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
    for profile, shape in [
        ("pooled", (8, 200)),
        ("temporal", (8, 4, 200)),
        ("spatial", (8, 64, 200)),
        ("full", (8, 256, 200)),
    ]:
        model = VideoHead(profile)
        x = torch.randn(shape, requires_grad=True)
        y = model(x)
        assert y.shape == (8, 64)
        y.square().mean().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert sum(p.numel() for p in model.parameters()) < 20000


@pytest.mark.parametrize(
    "profiles,diagnostics_only,head_type",
    [
        (("pooled", "temporal"), False, "compact"),
        (("spatial", "full"), False, "compact"),
        (("full",), True, "compact"),
        (("full",), False, "feature-weights"),
        (("full",), False, "weighting-comparison"),
    ],
)
def test_pipeline(reference, tmp_path, monkeypatch, profiles, diagnostics_only, head_type):
    """Run real heads with synthetic frozen features and unchanged reserved-person exclusion."""
    cache, run = reference
    metadata = json.loads((run / "baseline.json").read_text())
    metadata["metrics"] = {}
    (run / "baseline.json").write_text(json.dumps(metadata))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"test")

    def extract(raw, checkpoint, device, batch_size, output, profiles):
        """Supply deterministic token features only for eligible development windows."""
        assert all(p != "reserved" for values in raw.values() for p, _ in values)
        rng = np.random.default_rng(2)
        banks = {profile: {s: {} for s in raw} for profile in profiles}
        for split, values in raw.items():
            for key in values:
                shapes = {"pooled": (200,), "temporal": (4, 200), "spatial": (64, 200), "full": (256, 200)}
                for profile in profiles:
                    banks[profile][split][key] = rng.normal(size=shapes[profile]).astype(np.float32)
        return banks, 0.0

    monkeypatch.setattr("hyperscanning_contrastive.video_projection.extract_banks", extract)
    result = run_video_projection(
        cache,
        run,
        tmp_path / "result",
        checkpoint,
        epochs=1,
        steps_per_epoch=2,
        pairs=4,
        profiles=profiles,
        diagnostics_only=diagnostics_only,
        head_type=head_type,
    )
    assert len(result["experiments"]) == (
        0 if diagnostics_only else 2 * len(profiles) * (3 if head_type == "weighting-comparison" else 1)
    )
    assert set(result["diagnostics"]) == set(profiles)
    if diagnostics_only:
        assert not list((tmp_path / "result").rglob("*.pt"))
    assert result["training_person_windows"] == 8
    for experiment in result["experiments"].values():
        assert experiment["arms"]["true"]["selected"]["epoch"] in (0, 1)
    assert (tmp_path / "result" / "comparison.png").is_file()
    if profiles == ("full",) and not diagnostics_only:
        import pandas as pd

        from hyperscanning_contrastive.data import sha256
        from hyperscanning_contrastive.video_cnn import load_bank
        from hyperscanning_contrastive.video_labram import load_reference
        from hyperscanning_contrastive.video_patch_scores import run_patch_scores

        _, _, windows = load_reference(cache, run)
        raw, _ = load_bank(cache, run, windows, standardize=False)
        banks, _ = extract(raw, checkpoint, "cpu", 16, tmp_path, profiles)
        indices = [(split, p, t) for split, values in banks["full"].items() for p, t in values]
        folder = tmp_path / "result"
        np.save(folder / "full_features.npy", np.stack([banks["full"][s][(p, t)] for s, p, t in indices]))
        pd.DataFrame(indices, columns=["split", "participant_id", "cache_start"]).to_csv(
            folder / "feature_windows.csv", index=False
        )
        for name in ("full_features.npy", "feature_windows.csv"):
            result["artifacts_sha256"][name] = sha256(folder / name)
        (folder / "projection.json").write_text(json.dumps(result))
        scored = run_patch_scores(folder, tmp_path / "scores")
        assert scored["flattened_reproduction_verified"]
        (folder / "full_global" / "centers.npz").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="mismatched source"):
            run_patch_scores(folder, tmp_path / "invalid")


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


def test_channel_and_time_order():
    """Structured heads preserve the intended channel-major temporal ordering during mixing."""
    for profile in ("spatial", "full"):
        model = VideoHead(profile)
        x = torch.randn(2, 64 if profile == "spatial" else 256, 200)
        z = torch.nn.functional.gelu(model.token(x))
        if profile == "spatial":
            expected = torch.einsum("rc,bcd->brd", model.channel_mix.weight, z)
        else:
            expected = torch.einsum("rc,bctd->brtd", model.channel_mix.weight, z.reshape(2, 64, 4, 16))
        torch.testing.assert_close(model(x), model.output(expected.flatten(1)))


def test_feature_weight_identity_and_learning():
    """Preserve direct cosine at initialization and learn a shared diagonal metric."""
    from hyperscanning_contrastive.contrastive import contrastive_loss

    torch.manual_seed(42)
    model = VideoHead("full", "feature-weights")
    a, b = torch.randn(2, 8, 256, 200)
    assert sum(p.numel() for p in model.parameters()) == 200
    torch.testing.assert_close(model(a), a.flatten(1), rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = contrastive_loss(model(a), model(b), 0.5)
    loss.backward()
    assert torch.isfinite(model.weight_logits.grad).all()
    assert model.weight_logits.grad.abs().sum() > 0
    optimizer.step()
    weights = model.feature_weights()
    assert (weights > 0).all() and weights.std() > 0
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    torch.testing.assert_close(model(a).reshape_as(a), a * weights)
    with pytest.raises(ValueError, match="full readout"):
        VideoHead("spatial", "feature-weights")


@pytest.mark.parametrize("kind,count", [("channel-weights", 64), ("combined-weights", 264)])
def test_channel_weight_order(kind, count):
    """Channel weights repeat across time and combine multiplicatively with feature weights."""
    model = VideoHead("full", kind)
    x = torch.randn(4, 256, 200)
    torch.testing.assert_close(model(x), x.flatten(1), rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == count
    with torch.no_grad():
        model.channel_logits.copy_(torch.linspace(-1, 1, 64))
        if kind == "combined-weights":
            model.weight_logits.copy_(torch.linspace(-0.5, 0.5, 200))
    expected = x.reshape(4, 64, 4, 200) * model.channel_weights()[None, :, None, None]
    if kind == "combined-weights":
        expected = expected * model.feature_weights()
    torch.testing.assert_close(model(x), expected.flatten(1))
    from hyperscanning_contrastive.contrastive import contrastive_loss

    loss = contrastive_loss(model(x[:2]), model(x[2:]), 0.5)
    loss.backward()
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
