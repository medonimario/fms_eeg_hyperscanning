"""Optional W&B metrics logging with one independently finished run per fold."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from skorch.callbacks import Callback

from .monitoring import CURVE_KEYS


@contextmanager
def tracked_run(config: dict, output: Path) -> Iterator[Any]:
    """Start a fold run only when enabled and mark interrupted/failed runs accordingly."""
    options = config["options"]
    mode = options.get("wandb_mode", "disabled")
    if mode == "disabled":
        yield None
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B logging requires: pip install wandb") from error
    group = options.get("wandb_group") or output.parent.name
    run = wandb.init(
        project=options.get("wandb_project", "labram-classification"),
        entity=options.get("wandb_entity"),
        group=group,
        name=f"{group}-fold-{config['fold']}",
        job_type="smoke" if options["smoke_windows_per_class"] else config.get("model", "finetune"),
        mode=mode,
        dir=str(output.resolve()),
        config=config,
    )
    try:
        run.define_metric("epoch")
        run.define_metric("epoch/*", step_metric="epoch")
        run.define_metric("optimizer_step")
        run.define_metric("update/*", step_metric="optimizer_step")
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


class WandbProgress(Callback):
    """Log accumulated training loss during epochs and validation scores at epoch end."""

    def __init__(self, run: Any, log_every: int = 20) -> None:
        """Bind an existing fold run and the optimizer-update logging interval."""
        self.run = run
        self.log_every = log_every

    def initialize(self) -> WandbProgress:
        """Reset update counters for a fresh training run."""
        self.updates = 0
        self.loss_sum = 0.0
        return self

    def on_batch_end(self, net: Any, training: bool, loss: Any, **kwargs: Any) -> None:
        """Log the mean loss of a complete accumulation group, never validation batches."""
        if not training:
            return
        self.loss_sum += float(loss.item())
        if net.microbatch_ % net.update_freq:
            return
        self.updates += 1
        if self.updates % self.log_every == 0 or net.microbatch_ == len(net.epoch_rates_) * net.update_freq:
            rates = [group["lr"] for group in net.optimizer_.param_groups]
            self.run.log(
                {
                    "optimizer_step": self.updates,
                    "update/train_loss": self.loss_sum / net.update_freq,
                    "update/lr_max": max(rates),
                    "update/lr_min": min(rates),
                }
            )
        self.loss_sum = 0.0

    def on_epoch_end(self, net: Any, **kwargs: Any) -> None:
        """Read completed skorch scores after its scoring callbacks have run."""
        row = net.history[-1]
        values = {
            f"epoch/{key}": float(row[key])
            for key in ("train_loss", *CURVE_KEYS, "dur", "diagnostic_duration")
            if key in row
        }
        self.run.log({"epoch": row["epoch"], **values})


def log_result(run: Any, result: dict) -> None:
    """Log final validation-selected performance, keeping test evaluation at the end."""
    if run is None:
        return
    values = {
        f"final/{role}/{key}": value
        for role in ("train", "valid", "test")
        if role in result
        for key, value in result[role].items()
        if key != "confusion_matrix"
    }
    for key in ("best_epoch", "best_C"):
        if key in result:
            values[key] = result[key]
    run.log(values)
    run.summary.update(values)
    for role in ("valid", "test"):
        run.summary[f"final/{role}/confusion_matrix"] = result[role]["confusion_matrix"]
