"""Development-only tiny-set EEG fitting with a shared LaBraM encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .contrastive import contrastive_loss, retrieval
from .data import CHANNELS, sha256


def load_tiny_batch(
    cache: Path, pairs: int = 8, fold: int = 0, group: int | None = None, condition: str = "open"
) -> tuple[torch.Tensor, pd.DataFrame, dict]:
    """Verify cache provenance and select distinct trials at offset zero within one development dyad.

    Returns:
        Unscaled microvolt signals, selected manifest rows, and completion metadata.
    """
    if pairs < 2:
        raise ValueError("At least two pairs are required.")
    metadata = json.loads((cache / "cache.json").read_text())
    expected = {
        "format_version": 1,
        "dataset": "circling-local-project1",
        "operation": "prepare",
        "channels": CHANNELS,
        "sfreq": 200,
        "units": "microvolts",
        "model_input_divisor": 100,
        "window_seconds": 4,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Incompatible paired EEG cache contract.")
    if "windows.csv" not in metadata["manifest_sha256"]:
        raise ValueError("Cache is missing the windows manifest checksum.")
    for name, digest in metadata["manifest_sha256"].items():
        if Path(name).name != name or sha256(cache / name) != digest:
            raise ValueError(f"Cache manifest checksum mismatch: {name}")
    rows = pd.read_csv(cache / "windows.csv")
    rows = rows[(rows.fold == fold) & (rows.condition == condition) & (rows.crop_start_s == 0)]
    if group is not None:
        rows = rows[rows.group == group]
    selected = None
    for _, candidate in rows.groupby("group", sort=True):
        candidate = candidate.sort_values(["condition_trial", "array_index"]).drop_duplicates("trial_id")
        if len(candidate) >= pairs:
            selected = candidate.iloc[:pairs].copy()
            break
    if selected is None:
        raise ValueError("No development dyad has enough distinct clean trials at offset zero for this request.")
    if selected.array.nunique() != 1:
        raise ValueError("Expected one paired signal array for the selected dyad.")
    name = selected.array.iloc[0]
    if Path(name).name != name or sha256(cache / name) != metadata["array_sha256"].get(name):
        raise ValueError("Signal array checksum mismatch.")
    array = np.load(cache / name, mmap_mode="r", allow_pickle=False)
    if array.ndim != 4 or array.shape[1:] != (2, 64, 800) or array.dtype != np.float32:
        raise ValueError("Expected float32 paired EEG shaped (N, 2, 64, 800).")
    indices = selected.array_index.to_numpy()
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(array)):
        raise ValueError("Invalid cache row indices.")
    signal = np.array(array[indices], copy=True)
    if not np.isfinite(signal).all():
        raise ValueError("Nonfinite EEG signals.")
    return torch.from_numpy(signal), selected, metadata


class ContrastiveLabram(nn.Module):
    """Shared mean-pooling LaBraM and nonlinear projection for both participants."""

    def __init__(self, initialization: str, checkpoint: Path, train_backbone: bool = False) -> None:
        """Reuse Project 1's weight conversion without changing its classification implementation."""
        super().__init__()
        from labram_classification.model import load_model

        self.backbone = load_model(checkpoint, drop_path=0.0, initialization=initialization).backbone
        width = self.backbone.final_layer.in_features
        self.backbone.final_layer = nn.Identity()
        self.backbone.requires_grad_(train_backbone)
        self.projection = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 128))

    def features(self, microvolts: torch.Tensor) -> torch.Tensor:
        """Scale microvolts exactly once and preserve canonical channel identities."""
        if microvolts.ndim != 3 or microvolts.shape[1:] != (64, 800):
            raise ValueError("Expected (batch, 64, 800) microvolt signals.")
        return self.backbone(microvolts / 100.0, ch_names=CHANNELS)

    def forward(self, microvolts: torch.Tensor) -> torch.Tensor:
        """Project pooled features into the contrastive embedding space."""
        return self.projection(self.features(microvolts))


def run_overfit(
    cache: Path,
    output: Path,
    checkpoint: Path,
    initialization: str = "pretrained",
    train_backbone: bool = False,
    device: str = "cpu",
    pairs: int = 8,
    steps: int = 300,
    fold: int = 0,
    group: int | None = None,
    condition: str = "open",
    seed: int = 42,
) -> dict:
    """Fit fixed partner pairs on development trials; report memorization, never generalization."""
    if steps < 1:
        raise ValueError("Steps must be positive.")
    signals, selected, metadata = load_tiny_batch(cache, pairs, fold, group, condition)
    output.mkdir(parents=True, exist_ok=False)
    selected.to_csv(output / "selected_windows.csv", index=False)
    torch.manual_seed(seed)
    model = ContrastiveLabram(initialization, checkpoint, train_backbone).to(device)
    model.eval()
    signals = signals.to(device)
    first, second = signals[:, 0], signals[:, 1]
    if not train_backbone:
        with torch.no_grad():
            first, second = model.features(first), model.features(second)
        encoder = model.projection
    else:
        encoder = model
    optimizer = torch.optim.AdamW(
        [{"params": model.projection.parameters(), "lr": 1e-3}]
        + ([{"params": model.backbone.parameters(), "lr": 1e-5}] if train_backbone else []),
        weight_decay=0.0,
    )
    history = []
    for step in range(steps + 1):
        with torch.set_grad_enabled(step < steps):
            a, b = encoder(first), encoder(second)
            loss = contrastive_loss(a, b)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite contrastive loss.")
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                metrics = retrieval(a, b)
            history.append({"step": step, "loss": loss.item(), **metrics})
            print(f"Step {step}: loss={loss.item():.4f}, top1={metrics['top1']:.3f}", flush=True)
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite model gradients.")
            optimizer.step()
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    torch.save(model.state_dict(), output / "model.pt")
    report = {
        "scope": "Training-set memorization diagnostic only; no held-out inference",
        "initialization": initialization,
        "train_backbone": train_backbone,
        "checkpoint_sha256": sha256(checkpoint) if initialization == "pretrained" else None,
        "cache": str(cache.resolve()),
        "cache_sha256": sha256(cache / "cache.json"),
        "source_manifest_sha256": metadata["manifest_sha256"],
        "development_fold": fold,
        "group": int(selected.group.iloc[0]),
        "condition": condition,
        "pairs": pairs,
        "steps": steps,
        "seed": seed,
        "device": device,
        "torch_version": torch.__version__,
        "temperature": 0.1,
        "projection_lr": 1e-3,
        "backbone_lr": 1e-5 if train_backbone else None,
        "weight_decay": 0.0,
        "chance_top1": 1 / pairs,
        "initial": history[0],
        "final": history[-1],
        "passed": history[-1]["top1"] >= 0.99 and history[-1]["loss"] < history[0]["loss"] * 0.1,
    }
    (output / "overfit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
