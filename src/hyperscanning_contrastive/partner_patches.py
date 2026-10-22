"""Controlled frozen-token spatial, temporal, and full partner readout comparisons."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .data import CHANNELS, sha256
from .overfit import ContrastiveLabram
from .partner_windows import extract_features, fit_window_bank, plan_windows, validate_optimization
from .ssl import EEGCache, SSLConfig

PROFILES = ("pooled", "spatial", "temporal", "full")


class TokenExtractor(nn.Module):
    """Capture contextual patch tokens before pooling normalization without altering the backbone."""

    def __init__(self, model: ContrastiveLabram) -> None:
        """Wrap a frozen mean-pooling model and track reconstruction accuracy."""
        super().__init__()
        self.model = model
        self.max_reconstruction_error = 0.0
        if model.backbone.fc_norm is None:
            raise ValueError("Patch comparison requires the mean-pooling LaBraM profile.")

    def features(self, microvolts: torch.Tensor) -> torch.Tensor:
        """Return channel-major, time-minor tokens and check the original pooled output."""
        captured = []

        def capture(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            """Collect transformer output before the pooling-specific normalization."""
            captured.append(output)

        handle = self.model.backbone.norm.register_forward_hook(capture)
        try:
            pooled = self.model.features(microvolts)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[1:] != (257, 200):
            raise ValueError("Expected CLS plus 64 channels × four patches × 200 features.")
        tokens = captured[0][:, 1:, :]
        reconstructed = self.model.backbone.fc_norm(tokens.mean(1))
        torch.testing.assert_close(reconstructed, pooled, atol=1e-6, rtol=1e-5)
        self.max_reconstruction_error = max(self.max_reconstruction_error, float((reconstructed - pooled).abs().max()))
        if not torch.isfinite(tokens).all():
            raise ValueError("Nonfinite patch representations.")
        return tokens.reshape(-1, 64, 4, 200)


def reduce_tokens(tokens: torch.Tensor, profile: str, norm: nn.Module) -> torch.Tensor:
    """Preserve the requested axes, applying frozen normalization after any averaging."""
    if tokens.shape[1:] != (64, 4, 200):
        raise ValueError("Expected channel-major token grid (batch, 64, 4, 200).")
    if profile == "pooled":
        return norm(tokens.flatten(1, 2).mean(1))
    if profile == "spatial":
        return norm(tokens.mean(2))
    if profile == "temporal":
        return norm(tokens.mean(1))
    if profile == "full":
        return norm(tokens.flatten(1, 2))
    raise ValueError(f"Unknown patch profile: {profile}")


def run_partner_patches(
    cache_path: Path,
    output: Path,
    checkpoint: Path,
    device: str = "cpu",
    initialization: str = "pretrained",
    validation_trials: int = 8,
    epochs: int = 20,
    seed: int = 42,
    split_seed: int = 2042,
    permutation_seed: int = 1042,
    groups_per_update: int = 4,
    temperature: float = 0.05,
    projection_lr: float = 0.001,
    weight_decay: float = 0.01,
    profiles: tuple[str, ...] = PROFILES,
) -> dict:
    """Extract once, then train four matched open-only readouts and export comparisons."""
    validate_optimization(temperature, projection_lr, weight_decay)
    if not profiles or len(set(profiles)) != len(profiles) or not set(profiles) <= set(PROFILES):
        raise ValueError("Profiles must be unique, nonempty, and supported.")
    if groups_per_update < 1:
        raise ValueError("groups_per_update must be positive.")
    cache = EEGCache(cache_path, SSLConfig(initialization=initialization, pairs=2, epochs=epochs), all_windows=True)
    trials, rows, census = plan_windows(cache, validation_trials, split_seed, ("open",))
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = ContrastiveLabram(initialization, checkpoint, False).to(device).eval()
    extractor = TokenExtractor(model).eval()
    tokens = extract_features(extractor, cache, rows)
    norm = copy.deepcopy(model.backbone.fc_norm).cpu().eval()
    torch.save(norm.state_dict(), output / "frozen_fc_norm.pt")
    metadata = {
        "fc_norm_sha256": sha256(output / "frozen_fc_norm.pt"),
        "scope": "Exploratory open-only within-dyad patch readout comparison",
        "token_shape": [64, 4, 200],
        "channel_order": list(CHANNELS),
        "token_order": "channel-major, time-minor; CLS excluded",
        "token_source": "final contextual tokens before fc_norm",
        "normalization": "frozen fc_norm after axis reduction; position-wise training-only standardization",
        "max_pooled_reconstruction_error": extractor.max_reconstruction_error,
        "cache_sha256": sha256(cache_path / "cache.json"),
        "test_evaluated": False,
        "external_validation_evaluated": False,
        "capacity_warning": "Heads differ in parameter count; validation gains do not isolate capacity from structure",
    }
    del extractor, model
    (output / "token_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    reports = {}
    for profile in profiles:
        folder = output / profile
        folder.mkdir()
        for name, frame in (("trial_splits", trials), ("windows", rows), ("census", census)):
            frame.to_csv(folder / f"{name}.csv", index=False)
        with torch.no_grad():
            bank = {
                split: {
                    key: {**entry, **{member: reduce_tokens(entry[member], profile, norm) for member in ("a", "b")}}
                    for key, entry in entries.items()
                }
                for split, entries in tokens.items()
            }
        reports[profile] = fit_window_bank(
            cache_path,
            folder,
            checkpoint,
            bank,
            trials,
            rows,
            initialization,
            device,
            validation_trials,
            epochs,
            seed,
            split_seed,
            permutation_seed,
            groups_per_update,
            ("open",),
            structured=profile != "pooled",
            profile=profile,
            temperature=temperature,
            projection_lr=projection_lr,
            weight_decay=weight_decay,
        )
        del bank
    reference = reports[profiles[0]]["manifest_sha256"]
    if any(report["manifest_sha256"] != reference for report in reports.values()):
        raise ValueError("Readout splits, galleries, or shuffled mappings differ.")
    metadata["profiles"] = reports
    comparison = []
    for profile, report in reports.items():
        for arm, result in report["arms"].items():
            comparison.append(
                {
                    "profile": profile,
                    "arm": arm,
                    "parameters": report["trainable_parameters"],
                    "best_epoch": result["best_epoch"],
                    **result["macro"],
                }
            )
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    plot_comparison(output, profiles)
    (output / "patches.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def plot_comparison(output: Path, profiles: tuple[str, ...] = PROFILES) -> None:
    """Compare validation and training chance excess for each readout and control."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, len(profiles), figsize=(max(6, 4.25 * len(profiles)), 7), sharex=True, sharey="row", squeeze=False
    )
    for column, profile in enumerate(profiles):
        for arm, color in (("true", "#0072B2"), ("shuffled", "#D55E00")):
            history = pd.read_csv(output / profile / arm / "history.csv")
            axes[0, column].plot(history.epoch, history.mrr_excess, color=color, label=arm)
            axes[1, column].plot(history.epoch, history.train_objective_mrr_excess, color=color, label=arm)
        axes[0, column].set_title(profile)
        for row in (0, 1):
            axes[row, column].axhline(0, color="gray", linestyle=":")
            axes[row, column].grid(alpha=0.2)
            axes[row, column].legend()
        axes[1, column].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Validation MRR minus chance")
    axes[1, 0].set_ylabel("Training objective MRR minus chance")
    fig.suptitle("Open-only partner retrieval: retained spatial and temporal structure")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"patch_comparison.{suffix}", dpi=150)
    plt.close(fig)
