"""Small shared CNN contrastive pilot on verified video time splits."""

from __future__ import annotations

import copy
import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional

from .contrastive import contrastive_loss, retrieval
from .data import sha256
from .video_baseline import score_gallery
from .video_labram import load_reference


class VideoCNN(nn.Module):
    """Shared spatial mixing and temporal convolutions retaining ordered temporal bins."""

    def __init__(self) -> None:
        """Build a compact encoder without batch-dependent normalization or pretrained weights."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(64, 16, 1, bias=False),
            nn.Conv1d(16, 32, 25, stride=4, padding=12),
            nn.GELU(),
            nn.Conv1d(32, 32, 9, stride=4, padding=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(256, 64, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode channel-standardized windows without globally averaging their time axis."""
        return self.network(x)


def load_bank(cache: Path, run: Path, windows: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Load eligible development windows only and fit channel scales on training samples."""
    people = pd.read_csv(run / "participant_splits.csv")
    dev = set(people.loc[(people.split == "development") & (people.condition == "solo"), "participant_id"])
    source = pd.read_csv(cache / "windows.csv")
    source = source[(source.video_id == 0) & source.eligible & (source.condition == "solo")]
    source = source.set_index(["participant_id", "cache_start"])
    files = pd.read_csv(cache / "files.csv").dropna(subset=["array_path"]).set_index("array_path")
    bank = {"train": {}, "validation": {}}
    for person, rows in windows.reset_index().groupby("participant_id"):
        if person not in dev:
            continue
        rows = rows[rows.time_split.isin(bank)]
        path_name = rows.array_path.iloc[0]
        path = (cache / path_name).resolve()
        if not path.is_relative_to(cache) or sha256(path) != files.loc[path_name, "array_sha256"]:
            raise ValueError("Signal path/checksum mismatch.")
        signal = np.load(path, mmap_mode="r", allow_pickle=False)
        if signal.ndim != 2 or signal.shape[0] != 64 or signal.dtype != np.float32:
            raise ValueError("Invalid EEG array.")
        for row in rows.itertuples():
            original = source.loc[(person, row.cache_start)]
            if original.array_path != path_name or original.cache_stop != row.cache_stop:
                raise ValueError("Training window disagrees with cache manifest.")
            x = np.array(signal[:, row.cache_start : row.cache_stop])
            if x.shape != (64, 800) or not np.isfinite(x).all():
                raise ValueError("Invalid EEG window.")
            x -= x.mean(axis=1, keepdims=True)
            bank[row.time_split][(person, row.cache_start)] = x
    if not bank["train"]:
        raise ValueError("No training windows.")
    scale = np.sqrt(np.mean([np.mean(x.astype(np.float64) ** 2, axis=1) for x in bank["train"].values()], axis=0))
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    return {s: {k: x / scale[:, None] for k, x in values.items()} for s, values in bank.items()}, scale


def make_schedule(bank: dict, pairs: int, steps: int, seed: int) -> tuple[list, pd.DataFrame]:
    """Sample viewer pairs uniformly and unique common moments; retain identical arm inputs."""
    people = sorted({p for p, _ in bank})
    times = {p: {t for person, t in bank if person == p} for p in people}
    eligible, census = [], []
    for a, b in combinations(people, 2):
        common = sorted(times[a] & times[b])
        census.append(dict(first=a, second=b, common_moments=len(common), eligible=len(common) >= pairs))
        if len(common) >= pairs:
            eligible.append((a, b, common))
    if not eligible:
        raise ValueError(f"No viewer pair has {pairs} common training moments.")
    rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(steps):
        a, b, common = eligible[int(rng.integers(len(eligible)))]
        moments = sorted(map(int, rng.choice(common, size=pairs, replace=False)))
        shift = int(rng.integers(1, pairs))
        schedule.append((a, b, moments, shift))
    return schedule, pd.DataFrame(census)


def batch(bank: dict, entry: tuple, device: str, shuffled: bool = False) -> tuple:
    """Build two K-window batches; misalignment rotates whole target windows without fixed points."""
    a, b, moments, shift = entry
    x = torch.from_numpy(np.stack([bank[(a, t)] for t in moments])).to(device)
    target = np.roll(moments, shift) if shuffled else moments
    y = torch.from_numpy(np.stack([bank[(b, int(t))] for t in target])).to(device)
    return x, y


def evaluate(
    model: nn.Module, bank: dict, galleries: pd.DataFrame, device: str, temperature: float
) -> tuple[dict, pd.DataFrame]:
    """Evaluate the fixed bidirectional galleries and average incident pairs then people."""
    model.eval()
    keys = sorted({(p, t) for r in galleries.itertuples() for p in (r.first, r.second) for t in json.loads(r.starts)})
    features = {}
    with torch.inference_mode():
        for i in range(0, len(keys), 32):
            selected = keys[i : i + 32]
            z = model(torch.from_numpy(np.stack([bank[k] for k in selected])).to(device))
            if not torch.isfinite(z).all():
                raise ValueError("Nonfinite evaluation embeddings.")
            features.update(zip(selected, functional.normalize(z, dim=1).cpu().numpy(), strict=True))
    rows = []
    for r in galleries.itertuples():
        a, b = [np.stack([features[(p, t)] for t in json.loads(r.starts)]) for p in (r.first, r.second)]
        metrics = score_gallery(a, b)
        metrics["loss"] = float(contrastive_loss(torch.from_numpy(a), torch.from_numpy(b), temperature))
        rows.append(dict(first=r.first, second=r.second, **metrics))
    frame = pd.DataFrame(rows)
    identities = pd.concat(
        [
            frame.rename(columns={"first": "participant_id"}).drop(columns="second"),
            frame.rename(columns={"second": "participant_id"}).drop(columns="first"),
        ]
    )
    persons = identities.groupby("participant_id").mean(numeric_only=True)
    return persons.mean().to_dict(), persons


def run_video_cnn(
    cache: Path,
    run: Path,
    output: Path,
    device: str = "cpu",
    epochs: int = 20,
    steps_per_epoch: int = 100,
    pairs: int = 8,
    temperature: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    seed: int = 42,
    tiny_steps: int = 200,
) -> dict:
    """Train matched true/misaligned arms after a disposable tiny-set engineering check."""
    if min(epochs, steps_per_epoch, tiny_steps) < 1 or pairs < 2:
        raise ValueError("Positive epochs/steps and at least two candidates required.")
    if not np.isfinite([temperature, learning_rate, weight_decay]).all() or min(temperature, learning_rate) <= 0:
        raise ValueError("Invalid optimization settings.")
    if weight_decay < 0:
        raise ValueError("Negative weight decay.")
    cache, run = cache.resolve(), run.resolve()
    reference, galleries, windows = load_reference(cache, run)
    bank, scale = load_bank(cache, run, windows)
    schedule, census = make_schedule(bank["train"], pairs, epochs * steps_per_epoch, seed)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("participant_splits.csv", "window_splits.csv", "galleries.csv"):
        shutil.copyfile(run / name, output / name)
    census.to_csv(output / "training_pair_census.csv", index=False)
    pd.DataFrame(
        [
            dict(step=i, first=a, second=b, starts=json.dumps(t), target_rotation=s)
            for i, (a, b, t, s) in enumerate(schedule)
        ]
    ).to_csv(output / "schedule.csv", index=False)
    np.save(output / "channel_scale.npy", scale)
    torch.manual_seed(seed)
    initial = VideoCNN().to(device)
    torch.save(initial.state_dict(), output / "initial.pt")
    tiny = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=learning_rate, weight_decay=0)
    x, y = batch(bank["train"], schedule[0], device)
    tiny_history = []
    for step in range(tiny_steps + 1):
        a, b = tiny(x), tiny(y)
        loss = contrastive_loss(a, b, temperature)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite tiny-set loss.")
        tiny_history.append(dict(step=step, loss=loss.item(), **retrieval(a, b)))
        if step < tiny_steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    pd.DataFrame(tiny_history).to_csv(output / "tiny_history.csv", index=False)
    passed = tiny_history[-1]["top1"] >= 0.95
    (output / "tiny.json").write_text(json.dumps(dict(passed=passed, final=tiny_history[-1]), indent=2))
    if not passed:
        raise ValueError("Tiny-set fitting check failed; inspect tiny_history.csv before full training.")
    del tiny, optimizer
    histories, summaries = {}, {}
    for arm in ("true", "misaligned"):
        folder = output / arm
        folder.mkdir()
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        history, best, best_epoch, coverage = [], -np.inf, 0, set()
        for epoch in range(epochs + 1):
            losses, epoch_keys = [], set()
            if epoch:
                model.train()
                for entry in schedule[(epoch - 1) * steps_per_epoch : epoch * steps_per_epoch]:
                    a, b = batch(bank["train"], entry, device, arm == "misaligned")
                    loss = contrastive_loss(model(a), model(b), temperature)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss.")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    epoch_keys.update((p, t) for p in entry[:2] for t in entry[2])
                coverage.update(epoch_keys)
            row = dict(
                epoch=epoch,
                objective_loss=float(np.mean(losses)) if losses else None,
                epoch_unique_windows=len(epoch_keys),
                cumulative_unique_windows=len(coverage),
            )
            for split in ("train", "validation"):
                metrics, persons = evaluate(
                    model, bank[split], galleries[galleries.split == split], device, temperature
                )
                row.update({f"{split}_{k}": v for k, v in metrics.items()})
                if split == "validation" and metrics["mrr"] > best:
                    best, best_epoch = metrics["mrr"], epoch
                    torch.save(model.state_dict(), folder / "best.pt")
                    persons.to_csv(folder / "best_validation_people.csv")
            history.append(row)
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
            print(f"{arm} epoch {epoch}: validation MRR={row['validation_mrr']:.4f}", flush=True)
        torch.save(model.state_dict(), folder / "last.pt")
        histories[arm] = pd.DataFrame(history)
        summaries[arm] = dict(best_epoch=best_epoch, selected=history[best_epoch], final=history[-1])
    plot_history(histories, output, reference["metrics"])
    report = dict(
        scope="Solo video0 CNN held-time exploratory training; no significance test",
        cache_sha256=sha256(cache / "cache.json"),
        reference_sha256=sha256(run / "baseline.json"),
        seed=seed,
        device=device,
        torch_version=torch.__version__,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        training_candidates=pairs,
        negatives_per_query=pairs - 1,
        encoder_inputs_per_update=2 * pairs,
        temperature=temperature,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        parameters=sum(p.numel() for p in initial.parameters()),
        training_person_windows=len(bank["train"]),
        validation_person_windows=len(bank["validation"]),
        eligible_training_pairs=int(census.eligible.sum()),
        reserved_people_evaluated=False,
        preprocessing="Per-window channel demeaning; training-only global channel RMS scaling",
        control="Identical inputs/schedule; nonzero circular rotation of target batch at each step",
        training_spacing="Distinct nonoverlapping 4s moments; adjacent moments allowed",
        selection="Maximum validation MRR including epoch zero; ties retain earliest epoch",
        gallery_sha256=sha256(output / "galleries.csv"),
        chance_top1=reference["chance_top1"],
        chance_mrr=reference["chance_mrr"],
        baseline=reference["metrics"],
        arms=summaries,
        artifacts_sha256={str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()},
    )
    (output / "cnn.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def plot_history(histories: dict, output: Path, baselines: dict) -> None:
    """Plot loss and retrieval curves for both matched arms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for arm, frame in histories.items():
        axes[0].plot(frame.epoch, frame.objective_loss, label=f"{arm}: batch objective")
        for ax, metric in zip(axes[1:], ("loss", "top1", "mrr"), strict=True):
            for split, style in [("train", "--"), ("validation", "-")]:
                ax.plot(frame.epoch, frame[f"{split}_{metric}"], style, label=f"{arm}: {split}")
    for ax, title in zip(
        axes,
        (
            "Training objective (K candidates)",
            "Fixed-gallery loss (K=4)",
            "Fixed-gallery top1 (K=4)",
            "Fixed-gallery MRR (K=4)",
        ),
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    axes[2].axhline(0.25, color="gray", linestyle=":")
    axes[3].axhline(sum(1 / np.arange(1, 5)) / 4, color="gray", linestyle=":")
    for ax, metric in zip(axes[2:], ("top1", "mrr"), strict=True):
        for name, color in [("waveform", "purple"), ("corrca", "brown")]:
            value = baselines.get("validation", {}).get(name, {}).get(metric)
            if value is not None:
                ax.axhline(value, color=color, linestyle=":", label=f"{name}: validation")
        ax.legend(fontsize=6)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"learning_curves.{extension}", dpi=160)
    plt.close(fig)
