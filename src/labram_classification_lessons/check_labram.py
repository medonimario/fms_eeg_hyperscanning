"""Lesson 3: load pretrained LaBraM and check one real binary-classification batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from braindecode.models import Labram
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from torch import nn
from torch.utils.data import DataLoader, Subset

if __package__:
    from . import inspect_mirrorgame as preprocessing
    from .mirrorgame_dataset import build_datasets, make_manifest
else:
    import inspect_mirrorgame as preprocessing
    from mirrorgame_dataset import build_datasets, make_manifest

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "labram-base.pth"

class BinaryLabram(nn.Module):
    """Bind channel identities to native LaBraM and return one logit per window."""

    def __init__(self, backbone: Labram, ch_names: list[str]) -> None:
        """Validate the ordered input channels and register the pretrained backbone.

        Args:
            backbone: LaBraM with a single-output classification head.
            ch_names: Names in exactly the order of the input channel axis.

        Raises:
            ValueError: If channels are empty, duplicated, or unknown to LaBraM.
        """
        super().__init__()
        names = [name.upper() for name in ch_names]
        canonical = {name.upper() for name in LABRAM_CHANNEL_ORDER}
        if not names or len(names) != len(set(names)) or not set(names) <= canonical:
            raise ValueError("Channel names must be unique and belong to LaBraM's canonical channel list.")
        self.backbone = backbone
        self.ch_names = list(ch_names)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Return raw logits with shape (batch,), suitable for BCEWithLogitsLoss.

        Args:
            signal: Already scaled float tensor shaped (batch, channels, 800).

        Returns:
            One unbounded classification score per window.

        Raises:
            ValueError: If the input shape does not match the four-second contract.
        """
        if signal.ndim != 3 or signal.shape[1:] != (len(self.ch_names), preprocessing.WINDOW_SAMPLES):
            raise ValueError("Expected (batch, configured channels, 800) model input.")
        return self.backbone(signal, ch_names=self.ch_names).squeeze(-1)


def read_student_weights(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Read the original student backbone without unrestricted pickle loading.

    Args:
        checkpoint: Original LaBraM pretraining checkpoint containing a model dictionary.

    Returns:
        Student parameter dictionary with the student prefix removed.

    Raises:
        ValueError: If the expected student backbone is absent.
    """
    allowed = [
        argparse.Namespace, np.dtype, type(np.dtype("float64")),
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
    ]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    student = {name.removeprefix("student."): value for name, value in saved["model"].items()
               if name.startswith("student.")}
    if not student:
        raise ValueError("Expected student.* weights in checkpoint['model'].")
    return student


def convert_student_weights(student: dict[str, torch.Tensor], n_time_entries: int) -> dict[str, torch.Tensor]:
    """Translate original LaBraM parameter names to Braindecode names.

    Args:
        student: Original student backbone weights, without the student prefix.
        n_time_entries: Temporal-table length required by the target model.

    Returns:
        Headless mean-pooling backbone weights for strict loading.
    """
    converted = {}
    for name, value in student.items():
        if name in {"mask_token", "lm_head.weight", "lm_head.bias"}:
            continue
        if name == "pos_embed":
            name = "position_embedding"
        elif name == "time_embed":
            name = "temporal_embedding"
            value = value[:, :n_time_entries].clone()
        elif name.startswith("patch_embed."):
            name = name.replace("patch_embed.", "patch_embed.temporal_conv.", 1)
        elif name.startswith("norm."):
            name = name.replace("norm.", "fc_norm.", 1)
        name = name.replace(".mlp.fc1.", ".mlp.0.").replace(".mlp.fc2.", ".mlp.2.")
        converted[name] = value
    return converted


def load_model(ch_names: list[str], checkpoint: Path = DEFAULT_CHECKPOINT) -> BinaryLabram:
    """Load the local pretrained backbone and add a new binary classification head.

    Args:
        ch_names: Ordered EEG channel names.
        checkpoint: Original local labram-base.pth file; no network access is used.

    Returns:
        Trainable binary model with mean pooling and verified backbone weights.
    """
    student = read_student_weights(checkpoint)
    backbone = Labram(n_times=800, n_chans=128, n_outputs=0, sfreq=200, use_mean_pooling=True)
    state = convert_student_weights(student, backbone.temporal_embedding.shape[1])
    backbone.load_state_dict(state, strict=True)
    backbone.reset_head(1)
    nn.init.trunc_normal_(backbone.final_layer.weight, std=0.02)
    nn.init.zeros_(backbone.final_layer.bias)
    with torch.no_grad():
        backbone.final_layer.weight.mul_(0.001)
    return BinaryLabram(backbone, ch_names)


def main() -> None:
    """Run forward and backward checks on training data without an optimizer step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", choices=preprocessing.CONTRASTS, default=preprocessing.DEFAULT_CONTRAST)
    parser.add_argument("--fold", type=int, choices=range(10), default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    manifest = make_manifest(preprocessing.get_data_directory(), args.contrast, args.fold)
    training = manifest[manifest.split == "train"]
    preview = training[training.dyad == training.dyad.min()]
    dataset = build_datasets(preview)["train"]
    channels = dataset.datasets[0].windows.ch_names
    if any(child.windows.ch_names != channels for child in dataset.datasets):
        raise ValueError("All windows must use the same ordered channels.")
    targets = dataset.get_metadata().target.to_numpy()
    indices = [int(np.flatnonzero(targets == label)[0]) for label in (0, 1)]
    batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=2, shuffle=False, num_workers=0)))
    signal, target, _ = batch
    signal = signal.to(args.device)
    target = target.to(device=args.device, dtype=torch.float32)
    model = load_model(channels, args.checkpoint).to(args.device)
    model.eval()
    logits = model(signal)
    if logits.shape != target.shape or not torch.isfinite(logits).all():
        raise ValueError("Expected finite logits with the same shape as the targets.")
    loss = nn.BCEWithLogitsLoss()(logits, target)
    if not torch.isfinite(loss):
        raise ValueError("The binary loss is not finite.")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("Expected finite gradients throughout the model.")
    print(f"Checkpoint: {args.checkpoint.resolve()} (strict backbone load)")
    print("Adaptations: 4-patch temporal table; mean pooling using loaded normalization; new single-logit head")
    print(f"Input: {tuple(signal.shape)}, {signal.dtype}; targets: {target.tolist()}")
    print(f"Logits: {tuple(logits.shape)}; values: {logits.detach().cpu().tolist()}")
    print(f"P(class 1): {logits.detach().sigmoid().cpu().tolist()}")
    print(f"BCE loss: {loss.item():.6f}; finite gradients verified")
    print("No optimizer step performed. The new head is untrained; these probabilities are not performance results.")


if __name__ == "__main__":
    main()
