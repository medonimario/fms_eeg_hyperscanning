"""Continuous movement encoding and controlled EEG temporal-dependency experiments."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional

from .contrastive import paired_logits, retrieval_metrics, symmetric_loss
from .plotting import plot_training_history
from .tiny_fit import digest, require
from .training import FeatureBank, epoch_schedule, summarize

OFFSETS = tuple(range(0, 15, 2))
PROFILES = ("linear", "temporal", "pointwise")


class ResidualBlock(nn.Module):
    """Mix neighboring positions or apply an exactly parameter-matched pointwise MLP."""

    def __init__(self, kind: str, dilation: int, dropout: float) -> None:
        """Use 3,072 mixing weights and 64 normalization parameters in either branch."""
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.kind = kind
        if kind == "temporal":
            self.mix = nn.Conv1d(32, 32, 3, padding=dilation, dilation=dilation, bias=False)
        else:
            require(kind == "pointwise", "Invalid residual block")
            self.mix = nn.Sequential(nn.Linear(32, 48, bias=False), nn.GELU(), nn.Linear(48, 32, bias=False))
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return residual sequence in batch/time/channel order, without crossing trials."""
        value = self.norm(sequence)
        if self.kind == "temporal":
            value = self.mix(value.transpose(1, 2)).transpose(1, 2)
        else:
            value = self.mix(value)
        return sequence + self.dropout(functional.gelu(value))


class OrderedReadout(nn.Module):
    """Pool into four ordered bins and map their concatenation to one trial embedding."""

    def __init__(self) -> None:
        """Initialize a bias-free 128-to-32 trial map."""
        super().__init__()
        self.projection = nn.Linear(128, 32, bias=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Preserve bin order rather than globally averaging away trial progression."""
        bins = functional.adaptive_avg_pool1d(sequence.transpose(1, 2), 4).transpose(1, 2)
        return self.projection(bins.flatten(1))


class TrialModel(nn.Module):
    """Encode eight frozen EEG readouts and a complete 2-by-900 movement trajectory."""

    def __init__(self, profile: str, dropout: float = 0.1) -> None:
        """Build shared movement CNN and one of three controlled EEG branches."""
        super().__init__()
        require(profile in PROFILES, "Invalid EEG profile")
        self.eeg_projection = nn.Linear(800, 32, bias=False)
        self.eeg_readout = OrderedReadout()
        self.movement_stem = nn.Sequential(
            nn.Conv1d(2, 16, 9, stride=5, padding=4, bias=False),
            nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=3, padding=3, bias=False),
            nn.GELU(),
        )
        self.movement_blocks = nn.Sequential(*[ResidualBlock("temporal", d, dropout) for d in (1, 2, 4)])
        self.movement_readout = OrderedReadout()
        self.eeg_blocks = nn.Sequential(
            *([] if profile == "linear" else [ResidualBlock(profile, d, dropout) for d in (1, 2, 4)])
        )

    def forward(self, eeg: torch.Tensor, movement: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce unnormalized trial vectors; the shared cosine scorer normalizes them."""
        require(eeg.ndim == 3 and eeg.shape[1:] == (8, 800), "Expected eight temporal EEG readouts")
        require(movement.shape == (len(eeg), 2, 900), "Expected continuous 18-second movement")
        encoded_eeg = self.eeg_readout(self.eeg_blocks(self.eeg_projection(eeg)))
        sequence = self.movement_stem(movement).transpose(1, 2)
        encoded_movement = self.movement_readout(self.movement_blocks(sequence))
        return encoded_eeg, encoded_movement


def build_sequences(bank: FeatureBank, seed: int) -> tuple[list[dict], dict]:
    """Load unique full movement rows and fit channel means/scalar RMS on training time samples."""
    windows, statistics = bank.build("temporal", seed, OFFSETS)
    grouped = {}
    for g in windows:
        grouped.setdefault((g["group"], g["member"], g["split"]), []).append(g)
    sets = []
    for (group, member, split), rows in grouped.items():
        rows.sort(key=lambda g: g["offset"])
        require(tuple(g["offset"] for g in rows) == OFFSETS, "Missing EEG offsets")
        first = rows[0]
        require(all(g["trial_ids"] == first["trial_ids"] for g in rows), "EEG trial order mismatch")
        require(all(torch.equal(g["mapping"], first["mapping"]) for g in rows), "Incoherent shuffle")
        trials = bank.trials[bank.trials.group == group]
        indices = trials.set_index("trial_id").loc[first["trial_ids"]].movement_feature_index.to_numpy()
        array = bank.array(f"{group}_movement.npy", (len(trials), 2, 2, 900))
        movement = torch.from_numpy(np.array(array[indices, "ab".index(member)], copy=True))
        require(bool(torch.isfinite(movement).all()), "Nonfinite movement")
        sets.append(
            dict(
                gallery_id=first["gallery_id"].rsplit(":", 1)[0],
                group=group,
                member=member,
                split=split,
                trial_ids=first["trial_ids"],
                conditions=first["conditions"],
                mapping=first["mapping"],
                eeg=torch.stack([g["eeg"] for g in rows], dim=1),
                movement=movement,
            )
        )
    for person, stats in statistics.items():
        rows = [g for g in sets if f"{g['group']}:{g['member']}" == person]
        training = torch.cat([g["movement"] for g in rows if g["split"] == "train"])
        mean = training.mean(dim=(0, 2), keepdim=True)
        scale = (training - mean).square().mean().sqrt().clamp_min(1e-8)
        stats.pop("movement")
        stats["continuous_movement"] = dict(mean=mean, scale=scale)
        for g in rows:
            g["movement"] = (g["movement"] - mean) / scale
    return sets, statistics


def score(model: TrialModel, gallery: dict, device: str, temperature: float) -> torch.Tensor:
    """Compute trial-to-trial cosine logits from independently encoded modalities."""
    eeg, movement = model(gallery["eeg"].to(device), gallery["movement"].to(device))
    return paired_logits(eeg, movement, temperature)


@torch.no_grad()
def evaluate(model: TrialModel, sets: list[dict], device: str, temperature: float, arm: str) -> pd.DataFrame:
    """Evaluate true paired trial ranks, including condition-restricted galleries."""
    model.eval()
    records = []
    for g in sets:
        logits = score(model, g, device, temperature)
        mapping = g["mapping"].to(device) if arm == "shuffled" and g["split"] == "train" else None
        loss = float(symmetric_loss(logits, mapping))
        for condition in ("all", "closed1", "closed2"):
            indices = [i for i, c in enumerate(g["conditions"]) if condition == "all" or c == condition]
            if not indices:
                continue
            ix = torch.tensor(indices, device=device)
            for scope in ("mixed",) if condition == "all" else ("mixed", "within-condition"):
                forward, reverse, positives = logits[ix], logits.T[ix], ix
                if scope == "within-condition":
                    forward, reverse = forward[:, ix], reverse[:, ix]
                    positives = torch.arange(len(ix), device=device)
                metrics = retrieval_metrics(forward, positives)
                backward = retrieval_metrics(reverse, positives)
                records.append(
                    dict(
                        gallery_id=g["gallery_id"],
                        group=g["group"],
                        member=g["member"],
                        split=g["split"],
                        condition=condition,
                        scope=scope,
                        candidates=forward.shape[1],
                        queries=len(ix),
                        assigned_loss=loss,
                        reverse_top1=backward["top1"],
                        reverse_mrr=backward["mrr"],
                        **metrics,
                    )
                )
    return pd.DataFrame(records)


def fit(
    initial: TrialModel,
    sets: list[dict],
    destination: Path,
    profile: str,
    arm: str,
    schedule: list[list[int]],
    seed: int,
    device: str,
    temperature: float,
    learning_rate: float,
    weight_decay: float,
    groups_per_update: int,
) -> dict:
    """Fit one arm from matched weights/RNG, selecting validation MRR including epoch zero."""
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    training = [g for g in sets if g["split"] == "train"]
    history, best, updates = [], -math.inf, 0
    for epoch in range(len(schedule) + 1):
        model.train()
        gradient_norm = 0.0
        if epoch:
            order = schedule[epoch - 1]
            for start in range(0, len(order), groups_per_update):
                chunk = order[start : start + groups_per_update]
                optimizer.zero_grad()
                for index in chunk:
                    g = training[index]
                    mapping = g["mapping"].to(device) if arm == "shuffled" else None
                    loss = symmetric_loss(score(model, g, device, temperature), mapping)
                    require(bool(torch.isfinite(loss)), "Nonfinite loss")
                    (loss / len(chunk)).backward()
                require(
                    all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                    "Invalid gradients",
                )
                gradient_norm = max(gradient_norm, float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5)))
                optimizer.step()
                updates += 1
        detail = evaluate(model, sets, device, temperature, arm)
        metrics = summarize(detail[detail.scope == "mixed"])
        row = dict(epoch=epoch, updates=updates, gradient_norm_max=gradient_norm, **metrics)
        history.append(row)
        pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
        plot_training_history(pd.DataFrame(history), destination, f"{arm}; continuous movement")
        print(f"{profile}/{arm} epoch {epoch}: validation MRR={metrics['validation_mrr']:.4f}", flush=True)
        if metrics["validation_mrr"] > best:
            best = metrics["validation_mrr"]
            selected = dict(profile=profile, arm=arm, best_epoch=epoch, **row)
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, destination / "best_model.pt"
            )
            detail.to_csv(destination / "best_gallery_metrics.csv", index=False)
            with torch.no_grad():
                scores = {g["gallery_id"]: score(model, g, device, temperature).cpu().numpy() for g in sets}
            np.savez(destination / "best_scores.npz", **scores)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, destination / "last_model.pt")
    detail.to_csv(destination / "last_gallery_metrics.csv", index=False)
    (destination / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def run_temporal_trial(
    features: Path,
    run: Path,
    output: Path,
    epochs: int = 20,
    seed: int = 42,
    device: str = "cpu",
    learning_rate: float = 0.0001,
    weight_decay: float = 0.1,
    temperature: float = 0.1,
    groups_per_update: int = 4,
) -> dict:
    """Compare three EEG branches with identical continuous movement architecture and hard CL."""
    require(epochs > 0 and seed >= 0 and groups_per_update > 0, "Invalid budget/seed")
    require(all(math.isfinite(v) for v in (learning_rate, weight_decay, temperature)), "Nonfinite settings")
    require(learning_rate > 0 and temperature > 0 and weight_decay >= 0, "Invalid optimization settings")
    bank = FeatureBank(features, run)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        operation="temporal-trial",
        format_version=1,
        features_source=str(features.resolve()),
        split_source=str(run.resolve()),
        features_sha256=digest(features / "features.json"),
        split_sha256=digest(run / "split.json"),
        profiles=list(PROFILES),
        epochs=epochs,
        seed=seed,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        groups_per_update=groups_per_update,
        offsets=list(OFFSETS),
        embedding_dim=32,
        ordered_bins=4,
        dropout=0.1,
        eeg_dilations=[1, 2, 4],
        movement_shape=[2, 900],
        movement_sfreq=50,
        movement_stem="Conv1d 2->16 k9/s5/p4; GELU; Conv1d 16->32 k7/s3/p3; GELU; no biases",
        movement_temporal="three residual k3 blocks, dilations 1/2/4; per-position LayerNorm/GELU/dropout",
        movement_normalization="participant training trials/time: two channel means, shared scalar RMS",
        eeg_normalization="participant training windows only, feature mean and scalar RMS",
        objective="symmetric hard-target trial cosine cross entropy; one candidate per trial",
        selection="dyad-macro validation MRR; epoch zero eligible; earliest tie; pair2020 excluded",
        test_features_accessed=False,
        reserved_features_accessed=False,
        scope="Exploratory known-person held-trial retrieval; frozen EEG backbone; bidirectional context",
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sets, stats = build_sequences(bank, seed + 1042)
    torch.save(stats, output / "normalization.pt")
    manifest = [
        {k: v for k, v in g.items() if not isinstance(v, torch.Tensor)}
        | {"shuffled_positive_columns": g["mapping"].tolist()}
        for g in sets
    ]
    (output / "galleries.json").write_text(json.dumps(manifest, indent=2) + "\n")
    schedule = epoch_schedule(sum(g["split"] == "train" for g in sets), epochs, seed + 2042)
    (output / "schedule.json").write_text(json.dumps(schedule) + "\n")
    comparison, counts = [], {}
    for profile in PROFILES:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            initial = TrialModel(profile)
        counts[profile] = sum(p.numel() for p in initial.parameters())
        folder = output / profile
        folder.mkdir()
        torch.save(initial.state_dict(), folder / "initial_model.pt")
        for arm in ("true", "shuffled"):
            comparison.append(
                fit(
                    initial,
                    sets,
                    folder / arm,
                    profile,
                    arm,
                    schedule,
                    seed + 3042,
                    device,
                    temperature,
                    learning_rate,
                    weight_decay,
                    groups_per_update,
                )
            )
            pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    result = dict(
        **config,
        complete=True,
        parameter_counts=counts,
        results=comparison,
        versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        input_array_sha256={n: bank.metadata["artifact_sha256"][n] for n in sorted(bank.verified)},
        code_sha256={
            n: digest(Path(__file__).with_name(n))
            for n in ("temporal_trial.py", "training.py", "tiny_fit.py", "contrastive.py", "plotting.py")
        },
        output_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    (output / "temporal_trial.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
