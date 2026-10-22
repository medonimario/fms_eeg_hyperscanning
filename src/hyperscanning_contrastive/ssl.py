"""Single-person augmented-view SSL with dyad-held-out fixed retrieval galleries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .contrastive import contrastive_loss, pair_scores
from .data import CHANNELS, sha256
from .overfit import ContrastiveLabram


@dataclass
class SSLConfig:
    """Locked pilot settings shared by training, validation, and later test evaluation."""

    initialization: str = "pretrained"
    freeze_backbone: bool = False
    validation_fold: int = 1
    test_fold: int = 2
    pairs: int = 8
    epochs: int = 20
    steps_per_epoch: int = 100
    seed: int = 42
    amplitude: float = 0.1
    noise: float = 0.02
    channel_drop: float = 0.05
    temperature: float = 0.1

    def validate(self) -> None:
        """Reject invalid schedules, overlapping splits, and destructive augmentation settings."""
        if self.initialization not in {"pretrained", "random"}:
            raise ValueError("Unknown initialization.")
        if self.validation_fold == self.test_fold or self.pairs < 2 or min(self.epochs, self.steps_per_epoch) < 1:
            raise ValueError("Require distinct held-out folds, at least two pairs, and positive training lengths.")
        if not 0 <= self.amplitude < 1 or not 0 <= self.noise <= 1 or not 0 <= self.channel_drop < 1:
            raise ValueError("Invalid augmentation strengths.")
        if not 0 < self.temperature < float("inf"):
            raise ValueError("Temperature must be finite and positive.")


class EEGCache:
    """Verified memory-mapped paired cache with offset-matched, single-person galleries."""

    def __init__(self, path: Path, config: SSLConfig, all_windows: bool = False) -> None:
        """Verify manifests and signal hashes before selecting samples.

        Args:
            path: Prepared cache directory.
            config: Legacy gallery and split settings.
            all_windows: Retain the full verified manifest without constructing offset-zero strata.
        """
        config.validate()
        self.path = path
        self.metadata = json.loads((path / "cache.json").read_text())
        contract = {
            "format_version": 1,
            "operation": "prepare",
            "dataset": "circling-local-project1",
            "channels": CHANNELS,
            "sfreq": 200,
            "units": "microvolts",
            "model_input_divisor": 100,
            "window_seconds": 4,
        }
        if any(self.metadata.get(k) != v for k, v in contract.items()):
            raise ValueError("Incompatible prepared cache.")
        hashes = self.metadata["manifest_sha256"]
        if "windows.csv" not in hashes:
            raise ValueError("Missing windows checksum.")
        for name, digest in hashes.items():
            if Path(name).name != name or sha256(path / name) != digest:
                raise ValueError(f"Manifest checksum mismatch: {name}")
        rows = pd.read_csv(path / "windows.csv")
        if rows.groupby("group").fold.nunique().max() != 1:
            raise ValueError("A dyad occurs in multiple folds.")
        for group, frame in rows.groupby("group"):
            if self.metadata["group_folds"].get(str(group)) != int(frame.fold.iloc[0]):
                raise ValueError("Manifest fold assignment differs from cache metadata.")
        self.arrays = {}
        for name in sorted(rows.array.unique()):
            if Path(name).name != name or sha256(path / name) != self.metadata["array_sha256"].get(name):
                raise ValueError(f"Array checksum mismatch: {name}")
            array = np.load(path / name, mmap_mode="r", allow_pickle=False)
            if array.ndim != 4 or array.shape[1:] != (2, 64, 800) or array.dtype != np.float32:
                raise ValueError("Invalid paired signal shape/dtype.")
            indices = rows.loc[rows.array == name, "array_index"].to_numpy()
            if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(array)):
                raise ValueError("Invalid signal row indices.")
            self.arrays[name] = array
        if all_windows:
            self.rows = rows
            self.strata = {}
            return
        rows = rows[rows.crop_start_s == 0].copy()
        if rows.duplicated(["group", "condition", "trial_id"]).any():
            raise ValueError("Duplicate trials at the selected offset.")
        rows["split"] = "train"
        rows.loc[rows.fold == config.validation_fold, "split"] = "validation"
        rows.loc[rows.fold == config.test_fold, "split"] = "test"
        self.rows = rows
        self.strata: dict[str, list[pd.DataFrame]] = {name: [] for name in ("train", "validation", "test")}
        for (split, _, _), frame in rows.groupby(["split", "group", "condition"], sort=True):
            if len(frame) < config.pairs:
                raise ValueError("A dyad/condition has too few offset-zero trials; reduce pairs for all runs.")
            for member in (0, 1):
                self.strata[split].append(frame.sort_values("condition_trial").assign(member=member))
        if any(not frames for frames in self.strata.values()):
            raise ValueError("Train, validation and test must each contain eligible dyads.")
        for split, frames in self.strata.items():
            for group in {int(f.group.iloc[0]) for f in frames}:
                present = {f.condition.iloc[0] for f in frames if int(f.group.iloc[0]) == group}
                if present != set(self.metadata["conditions"]):
                    raise ValueError(f"Incomplete condition coverage for {split} dyad {group}.")

    def signals(self, rows: pd.DataFrame) -> torch.Tensor:
        """Load a single participant's unscaled windows; preserve sample and channel alignment."""
        values = np.stack([self.arrays[r.array][r.array_index, r.member] for r in rows.itertuples()])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite cache signals.")
        return torch.from_numpy(values)

    def galleries(self, split: str, config: SSLConfig) -> list[pd.DataFrame]:
        """Choose fixed signal-independent candidate sets, reproducible across initializations."""
        rng = np.random.default_rng(config.seed + 1000)
        return [frame.iloc[rng.choice(len(frame), config.pairs, replace=False)].copy() for frame in self.strata[split]]


def augment(signal: torch.Tensor, config: SSLConfig, generator: torch.Generator) -> torch.Tensor:
    """Apply independent amplitude, RMS-relative noise, and channel masks without moving time samples."""
    shape = (len(signal), 1, 1)
    scale = 1 + config.amplitude * (2 * torch.rand(shape, generator=generator) - 1)
    noise = (
        torch.randn(signal.shape, generator=generator) * signal.square().mean(-1, keepdim=True).sqrt() * config.noise
    )
    keep = torch.rand((len(signal), signal.shape[1], 1), generator=generator) >= config.channel_drop
    return (signal * scale + noise) * keep


def gallery_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    """Compute bidirectional retrieval with exact random-tie expectations, including MRR."""
    scores = pair_scores(first, second)
    positive = scores.diagonal()
    greater = torch.cat([(s > positive[:, None]).sum(1) for s in (scores, scores.T)])
    equal = torch.cat([(s == positive[:, None]).sum(1) for s in (scores, scores.T)])
    harmonic = torch.cat([scores.new_zeros(1), (1 / torch.arange(1, len(scores) + 1, device=scores.device)).cumsum(0)])
    return {
        "top1": ((greater == 0).float() / equal).mean().item(),
        "recall5": ((min(5, len(scores)) - greater).clamp(min=0).minimum(equal) / equal).mean().item(),
        "mrr": ((harmonic[greater + equal] - harmonic[greater]) / equal).mean().item(),
        "mean_rank": (1 + greater + (equal - 1) / 2).mean().item(),
        "embedding_std": torch.cat([first, second]).std(0, unbiased=False).mean().item(),
    }


def evaluate(
    model: nn.Module,
    cache: EEGCache,
    galleries: list[pd.DataFrame],
    config: SSLConfig,
    device: str,
    pooled: bool = False,
) -> pd.DataFrame:
    """Evaluate fixed augmentation draws without modifying optimizer or training RNG state."""
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 2000)
    result = []
    with torch.no_grad():
        for frame in galleries:
            signal = cache.signals(frame)
            first, second = augment(signal, config, generator).to(device), augment(signal, config, generator).to(device)
            encode = model.features if pooled else model
            a, b = encode(first), encode(second)
            result.append(
                {
                    "group": int(frame.group.iloc[0]),
                    "condition": frame.condition.iloc[0],
                    "member": int(frame.member.iloc[0]),
                    "pairs": len(frame),
                    "loss": contrastive_loss(a, b, config.temperature).item(),
                    **gallery_metrics(a, b),
                }
            )
    return pd.DataFrame(result)


def macro(frame: pd.DataFrame) -> dict[str, float]:
    """Average members, then conditions, then dyads with equal weights at each level."""
    columns = ["loss", "top1", "recall5", "mrr", "mean_rank", "embedding_std"]
    return frame.groupby(["group", "condition"])[columns].mean().groupby("group").mean().mean().to_dict()


def train_ssl(cache_path: Path, output: Path, checkpoint: Path, config: SSLConfig, device: str = "cpu") -> dict:
    """Train single-person SSL and select solely by validation MRR; never evaluate test signals."""
    cache = EEGCache(cache_path, config)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    model = ContrastiveLabram(config.initialization, checkpoint, not config.freeze_backbone).to(device)
    galleries = {name: cache.galleries(name, config) for name in ("validation", "test")}
    for split, frames in galleries.items():
        pd.concat(frames).to_csv(output / f"{split}_galleries.csv", index=False)
    cache.rows[["group", "fold", "split"]].drop_duplicates().to_csv(output / "splits.csv", index=False)
    settings = {
        "config": asdict(config),
        "cache_sha256": sha256(cache_path / "cache.json"),
        "cache": str(cache_path.resolve()),
        "device": device,
        "torch_version": torch.__version__,
        "checkpoint_sha256": sha256(checkpoint) if config.initialization == "pretrained" else None,
        "backbone_lr": (1e-4 if config.initialization == "random" else 1e-5),
        "projection_lr": 1e-3,
        "weight_decay": 0.01,
        "scope": "single-person augmented-view retrieval; not hyperscanning evidence",
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    torch.save(model.state_dict(), output / "initial.pt")
    pooled = evaluate(model, cache, galleries["validation"], config, device, pooled=True)
    pooled.to_csv(output / "validation_initial_pooled.csv", index=False)
    baseline = evaluate(model, cache, galleries["validation"], config, device)
    baseline.to_csv(output / "validation_initial.csv", index=False)
    best, best_epoch = macro(baseline)["mrr"], 0
    torch.save(model.state_dict(), output / "best.pt")
    baseline.to_csv(output / "validation_best.csv", index=False)
    history = [{"epoch": 0, "train_loss": None, **macro(baseline)}]
    groups = [{"params": model.projection.parameters(), "lr": settings["projection_lr"]}]
    if not config.freeze_backbone:
        groups.append({"params": model.backbone.parameters(), "lr": settings["backbone_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator().manual_seed(config.seed + 3000)
    for epoch in range(1, config.epochs + 1):
        model.train()
        if config.freeze_backbone:
            model.backbone.eval()
        losses = []
        for _ in range(config.steps_per_epoch):
            frames = cache.strata["train"]
            frame = frames[int(rng.integers(len(frames)))]
            signal = cache.signals(frame.iloc[rng.choice(len(frame), config.pairs, replace=False)])
            a, b = (
                model(augment(signal, config, generator).to(device)),
                model(augment(signal, config, generator).to(device)),
            )
            loss = contrastive_loss(a, b, config.temperature)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        validation = evaluate(model, cache, galleries["validation"], config, device)
        metrics = macro(validation)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        validation.to_csv(output / f"validation_epoch_{epoch:03d}.csv", index=False)
        if metrics["mrr"] > best:
            best, best_epoch = metrics["mrr"], epoch
            torch.save(model.state_dict(), output / "best.pt")
            validation.to_csv(output / "validation_best.csv", index=False)
        print(f"Epoch {epoch}: train loss={np.mean(losses):.4f}, validation MRR={metrics['mrr']:.4f}", flush=True)
    report = {
        "settings_sha256": sha256(output / "settings.json"),
        "checkpoint_sha256": {name: sha256(output / f"{name}.pt") for name in ("initial", "best")},
        "gallery_sha256": {name: sha256(output / f"{name}_galleries.csv") for name in ("validation", "test")},
        "best_epoch": best_epoch,
        "validation_mrr": best,
        "initial_validation": macro(baseline),
        "initial_pooled_validation": macro(pooled),
        "test_evaluated": False,
        "chance_top1": 1 / config.pairs,
        "chance_mrr": sum(1 / i for i in range(1, config.pairs + 1)) / config.pairs,
    }
    (output / "ssl.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_evaluation(cache_path: Path, run: Path, split: str) -> tuple[EEGCache, SSLConfig, list[pd.DataFrame]]:
    """Verify completed run provenance and reconstruct a saved held-out gallery without scoring it."""
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test.")
    run = run.expanduser().resolve()
    if not (run / "ssl.json").is_file():
        raise ValueError(
            f"Missing Stage 2 completion marker: {run / 'ssl.json'}. "
            "--run must point to the completed circling_ssl_* directory containing ssl.json, "
            "settings.json, initial.pt and best.pt, not the cache, overfit run, or output directory."
        )
    completed = json.loads((run / "ssl.json").read_text())
    if sha256(run / "settings.json") != completed["settings_sha256"]:
        raise ValueError("Training settings changed after completion.")
    for name, digest in completed["checkpoint_sha256"].items():
        if name not in {"initial", "best"} or sha256(run / f"{name}.pt") != digest:
            raise ValueError("Model checksum mismatch.")
    if sha256(run / f"{split}_galleries.csv") != completed["gallery_sha256"][split]:
        raise ValueError("Saved galleries changed after training.")
    settings = json.loads((run / "settings.json").read_text())
    if sha256(cache_path / "cache.json") != settings["cache_sha256"]:
        raise ValueError("Evaluation cache differs from training cache.")
    config = SSLConfig(**settings["config"])
    cache = EEGCache(cache_path, config)
    galleries = cache.galleries(split, config)
    saved = pd.read_csv(run / f"{split}_galleries.csv")
    keys = ["group", "condition", "trial_id", "array", "array_index", "member", "crop_start_s", "split"]
    pd.testing.assert_frame_equal(
        saved[keys].reset_index(drop=True), pd.concat(galleries)[keys].reset_index(drop=True), check_dtype=False
    )
    return cache, config, galleries


def test_ssl(cache_path: Path, run: Path, output: Path, device: str = "cpu") -> dict:
    """Evaluate locked initial/best checkpoints on saved test galleries in a separate explicit step."""
    cache, config, galleries = load_evaluation(cache_path, run, "test")
    output.mkdir(parents=True, exist_ok=False)
    model = ContrastiveLabram("random", Path("unused"), False).to(device)
    result = {}
    for name in ("initial", "best"):
        model.load_state_dict(torch.load(run / f"{name}.pt", map_location="cpu", weights_only=True), strict=True)
        frame = evaluate(model, cache, galleries, config, device)
        frame.to_csv(output / f"{name}_test.csv", index=False)
        result[name] = macro(frame)
        if name == "initial":
            pooled = evaluate(model, cache, galleries, config, device, pooled=True)
            pooled.to_csv(output / "initial_pooled_test.csv", index=False)
            result["initial_pooled"] = macro(pooled)
    result["scope"] = "single-person augmented-view retrieval; not partner retrieval"
    (output / "test.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
