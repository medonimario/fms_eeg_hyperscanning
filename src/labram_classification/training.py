"""Braindecode training with the original layer decay and accumulation protocol."""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from braindecode import EEGClassifier
from skorch.callbacks import Checkpoint, EpochScoring
from skorch.dataset import unpack_data
from skorch.helper import predefined_split
from torch import nn
from torch.utils.data import Dataset

from .monitoring import EpochDiagnostics


class BinaryLoss(nn.BCEWithLogitsLoss):
    """Accept Braindecode's integer targets and compute stable float32 binary loss."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return mean binary cross entropy from one logit per example."""
        return super().forward(logits.float(), targets.float())


def parameter_groups(model: nn.Module, lr: float, decay: float, weight_decay: float) -> list[dict]:
    """Match the original 12-block LaBraM layer IDs and weight-decay exclusions."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        name = name.removeprefix("backbone.")
        if name in {"cls_token", "position_embedding"} or name.startswith("patch_embed."):
            layer = 0
        elif name.startswith("blocks."):
            layer = int(name.split(".")[1]) + 1
        else:
            layer = 13
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or name
            in {
                "cls_token",
                "position_embedding",
                "temporal_embedding",
            }
        )
        key = layer, no_decay
        if key not in groups:
            scale = decay ** (13 - layer)
            groups[key] = dict(params=[], lr=lr * scale, lr_scale=scale, weight_decay=0.0 if no_decay else weight_decay)
        groups[key]["params"].append(parameter)
    return list(groups.values())


def learning_rates(lr: float, minimum: float, epochs: int, warmup: int, steps: int) -> np.ndarray:
    """Reproduce the original inclusive linear warmup followed by cosine decay."""
    if steps < 1 or not 0 <= warmup < epochs:
        raise ValueError("Need at least one optimizer update per epoch and 0 <= warmup < epochs.")
    count = (epochs - warmup) * steps
    cosine = minimum + 0.5 * (lr - minimum) * (1 + np.cos(np.pi * np.arange(count) / count))
    return np.concatenate([np.linspace(0, lr, warmup * steps), cosine])


class FinetuneClassifier(EEGClassifier):
    """Keep EEGClassifier's fit/scoring API while matching accumulated AdamW updates."""

    def __init__(
        self,
        *args: Any,
        update_freq: int = 2,
        layer_decay: float = 0.65,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        amp: bool = True,
        **kwargs: Any,
    ) -> None:
        """Store optimizer protocol options alongside the standard classifier options."""
        self.update_freq = update_freq
        self.layer_decay = layer_decay
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.amp = amp
        super().__init__(*args, **kwargs)

    def initialize_optimizer(self, triggered_directly: bool = True) -> FinetuneClassifier:
        """Create layer-specific AdamW groups and a CUDA-only gradient scaler."""
        self.optimizer_ = torch.optim.AdamW(
            parameter_groups(self.module_, self.lr, self.layer_decay, self.weight_decay),
            lr=self.lr,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
        self.amp_enabled_ = self.amp and str(self.device).startswith("cuda")
        self.scaler_ = torch.amp.GradScaler("cuda", enabled=self.amp_enabled_)
        return self

    def infer(self, x: Any, **fit_params: Any) -> torch.Tensor:
        """Use CUDA autocast consistently for training, validation, and prediction."""
        with torch.autocast("cuda", enabled=self.amp_enabled_):
            return super().infer(x, **fit_params)

    def run_single_epoch(self, iterator: Any, training: bool, prefix: str, step_fn: Any, **fit_params: Any) -> None:
        """Drop incomplete accumulation groups, as in the original fine-tuning loop."""
        if training:
            steps = len(iterator.loader) // self.update_freq
            rates = learning_rates(self.lr, self.min_lr, self.max_epochs, self.warmup_epochs, steps)
            self.epoch_rates_ = rates[(len(self.history) - 1) * steps : len(self.history) * steps]
            self.microbatch_ = 0
            self.optimizer_.zero_grad(set_to_none=True)
            iterator = islice(iterator, steps * self.update_freq)
        super().run_single_epoch(iterator, training, prefix, step_fn, **fit_params)

    def train_step(self, batch: Any, **fit_params: Any) -> dict:
        """Backpropagate each microbatch and update once per complete accumulation group."""
        self._set_training(True)
        signal, target = unpack_data(batch)
        if self.microbatch_ % self.update_freq == 0:
            rate = self.epoch_rates_[self.microbatch_ // self.update_freq]
            for group in self.optimizer_.param_groups:
                group["lr"] = float(rate) * group["lr_scale"]
        prediction = self.infer(signal, **fit_params)
        loss = self.get_loss(prediction, target, X=signal, training=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        self.scaler_.scale(loss / self.update_freq).backward()
        self.microbatch_ += 1
        if self.microbatch_ % self.update_freq == 0:
            self.scaler_.step(self.optimizer_)
            self.scaler_.update()
            self.optimizer_.zero_grad(set_to_none=True)
        return dict(loss=loss.detach(), y_pred=prediction.detach())


def make_classifier(
    model: nn.Module,
    validation: Dataset,
    output: Path,
    options: dict,
    test: Dataset | None = None,
) -> FinetuneClassifier:
    """Select the best epoch by held-out validation balanced accuracy only."""
    scoring = EpochScoring("balanced_accuracy", name="valid_balanced_accuracy", lower_is_better=False)
    checkpoint = Checkpoint(
        monitor="valid_balanced_accuracy_best",
        dirname=str(output),
        f_params="best_model.pt",
        f_optimizer=None,
        f_criterion=None,
        f_history=None,
        load_best=True,
    )
    return FinetuneClassifier(
        module=model,
        criterion=BinaryLoss,
        classes=[0, 1],
        cropped=False,
        train_split=predefined_split(validation),
        iterator_train__shuffle=True,
        iterator_train__drop_last=True,
        iterator_train__num_workers=options["num_workers"],
        iterator_valid__num_workers=options["num_workers"],
        iterator_train__pin_memory=options["device"] == "cuda",
        iterator_valid__pin_memory=options["device"] == "cuda",
        callbacks=[
            ("balanced_accuracy", scoring),
            ("checkpoint", checkpoint),
            ("diagnostics", EpochDiagnostics(output, test)),
        ],
        max_epochs=options["epochs"],
        batch_size=options["batch_size"],
        lr=options["lr"],
        update_freq=options["update_freq"],
        layer_decay=options["layer_decay"],
        weight_decay=options["weight_decay"],
        warmup_epochs=options["warmup_epochs"],
        min_lr=options["min_lr"],
        amp=options["amp"],
        device=options["device"],
    )
