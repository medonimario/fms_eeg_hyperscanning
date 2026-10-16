"""Binary adapters for pretrained LaBraM and randomly initialized native EEGNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from braindecode.models import EEGNet, Labram
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from torch import nn

from .data import CHANNELS

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "labram-base.pth"


class BinaryEEGNet(nn.Module):
    """Adapt native EEGNet-8,2 to the shared single-logit classification interface."""

    def __init__(self, drop_prob: float = 0.5) -> None:
        """Initialize EEGNet from scratch for 64-channel, 200 Hz, four-second windows."""
        super().__init__()
        self.backbone = EEGNet(
            n_chans=len(CHANNELS),
            n_times=800,
            n_outputs=1,
            sfreq=200,
            F1=8,
            D=2,
            F2=16,
            kernel_length=64,
            depthwise_kernel_length=16,
            pool1_kernel_size=4,
            pool2_kernel_size=8,
            drop_prob=drop_prob,
            final_layer_with_constraint=True,
            conv_spatial_max_norm=1,
            norm_rate=0.25,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Return one raw logit per window, retaining the batch axis for batch size one."""
        if signal.ndim != 3 or signal.shape[1:] != (len(CHANNELS), 800):
            raise ValueError("Expected (batch, 64, 800) model input.")
        return self.backbone(signal).reshape(signal.shape[0])


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
        if signal.ndim != 3 or signal.shape[1:] != (len(self.ch_names), 800):
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
        argparse.Namespace,
        np.dtype,
        type(np.dtype("float64")),
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
    ]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    student = {
        name.removeprefix("student."): value for name, value in saved["model"].items() if name.startswith("student.")
    }
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
        if name in {"mask_token", "lm_head.weight", "lm_head.bias", "norm.weight", "norm.bias"}:
            continue
        if name == "pos_embed":
            name = "position_embedding"
        elif name == "time_embed":
            name = "temporal_embedding"
            value = value[:, :n_time_entries].clone()
        elif name.startswith("patch_embed."):
            name = name.replace("patch_embed.", "patch_embed.temporal_conv.", 1)
        name = name.replace(".mlp.fc1.", ".mlp.0.").replace(".mlp.fc2.", ".mlp.2.")
        converted[name] = value
    return converted


def load_model(checkpoint: Path = DEFAULT_CHECKPOINT, drop_path: float = 0.2) -> BinaryLabram:
    """Load the local pretrained backbone and add a new binary classification head.

    Args:
        drop_path: Stochastic depth rate used during fine-tuning.
        checkpoint: Original local labram-base.pth file; no network access is used.

    Returns:
        Trainable binary model with mean pooling and verified backbone weights.
    """
    student = read_student_weights(checkpoint)
    backbone = Labram(
        n_times=800,
        n_chans=128,
        n_outputs=0,
        sfreq=200,
        use_mean_pooling=True,
        drop_path_prob=drop_path,
        qkv_bias=False,
        use_abs_pos_emb=True,
    )
    state = convert_student_weights(student, backbone.temporal_embedding.shape[1])
    state.update({name: value for name, value in backbone.state_dict().items() if name.startswith("fc_norm.")})
    backbone.load_state_dict(state, strict=True)
    backbone.reset_head(1)
    nn.init.trunc_normal_(backbone.final_layer.weight, std=0.02)
    nn.init.zeros_(backbone.final_layer.bias)
    with torch.no_grad():
        backbone.final_layer.weight.mul_(0.001)
    return BinaryLabram(backbone, CHANNELS)
