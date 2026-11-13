"""Matched local/trial and hard/soft objectives on frozen four-second EEG features."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional

from .contrastive import retrieval_metrics
from .models import LinearHeads
from .plotting import plot_training_history
from .tiny_fit import center_scale, digest, require
from .training import FeatureBank, epoch_schedule, summarize

OFFSETS = tuple(range(0, 15, 2))
EXPERIMENTS = ("local-hard", "trial-hard", "local-soft", "trial-soft")


def build_trials(bank: FeatureBank, seed: int) -> tuple[list[dict], dict]:
    """Stack ordered windows and fit a separate full-trajectory teacher on training trials."""
    windows, stats = bank.build("temporal", seed, OFFSETS)
    grouped = {}
    for window in windows:
        key = (window["group"], window["member"], window["split"])
        grouped.setdefault(key, []).append(window)
    trials = []
    for (group, member, split), rows in grouped.items():
        rows.sort(key=lambda g: g["offset"])
        require(tuple(g["offset"] for g in rows) == OFFSETS, "Missing trial windows")
        first = rows[0]
        require(all(g["trial_ids"] == first["trial_ids"] for g in rows), "Trial order mismatch")
        require(all(torch.equal(g["mapping"], first["mapping"]) for g in rows), "Incoherent shuffle")
        eeg = torch.stack([g["eeg"] for g in rows], dim=1)
        movement = torch.stack([g["movement"] for g in rows], dim=1)
        fitted = stats[f"{group}:{member}"]["movement"]
        raw = (movement * fitted["scale"] + fitted["mean"]).reshape(len(movement), 8, 2, 200)
        require(torch.allclose(raw[:, :-1, :, 100:], raw[:, 1:, :, :100], atol=1e-5), "Overlap mismatch")
        trajectory = torch.cat([raw[:, 0]] + [raw[:, k, :, 100:] for k in range(1, 8)], dim=-1).flatten(1)
        trials.append(
            dict(
                gallery_id=first["gallery_id"].rsplit(":", 1)[0],
                group=group,
                member=member,
                split=split,
                trial_ids=first["trial_ids"],
                conditions=first["conditions"],
                mapping=first["mapping"],
                eeg=eeg,
                movement=movement,
                trajectory=trajectory,
            )
        )
    for person, fitted in stats.items():
        rows = [g for g in trials if f"{g['group']}:{g['member']}" == person]
        training = [g["trajectory"] for g in rows if g["split"] == "train"]
        _, teacher_stats = center_scale(torch.cat(training))
        fitted["trajectory_teacher"] = teacher_stats
        for g in rows:
            descriptor = (g.pop("trajectory") - teacher_stats["mean"]) / teacher_stats["scale"]
            descriptor = functional.normalize(descriptor, dim=-1)
            local = functional.normalize(g["movement"], dim=-1)
            g["trial_similarity"] = descriptor @ descriptor.T
            g["local_similarity"] = torch.einsum("ikd,jkd->kij", local, local)
    return trials, stats


def teacher_probabilities(similarity: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return row-normalized fixed movement similarities, including diagonal self-similarity."""
    return (similarity / temperature).softmax(dim=-1)


def calibrate_teacher(sets: list[dict], target: float = 2.0) -> tuple[dict, pd.DataFrame]:
    """Choose temperatures by training-only effective-positive counts, never retrieval scores."""
    require(math.isfinite(target) and target > 1, "Effective-positive target must exceed one")
    temperatures, records = {}, []
    for level in ("local", "trial"):
        training = [g for g in sets if g["split"] == "train"]
        require(all(len(g["trial_ids"]) > target for g in training), "Teacher target exceeds gallery capacity")
        candidates = np.geomspace(0.001, 10.0, 81)
        errors = []
        for temperature in candidates:
            effective = []
            for g in training:
                q = teacher_probabilities(g[f"{level}_similarity"], float(temperature))
                effective.append((-(q * q.clamp_min(1e-30).log()).sum(-1)).exp().flatten())
            median = float(torch.cat(effective).median())
            errors.append(abs(median - target))
        temperature = float(candidates[int(np.argmin(errors))])
        temperatures[level] = temperature
        for g in training:
            similarity = g[f"{level}_similarity"]
            q = teacher_probabilities(similarity, temperature)
            entropy = -(q * q.clamp_min(1e-30).log()).sum(-1)
            offdiag = similarity[..., ~torch.eye(len(g["trial_ids"]), dtype=torch.bool)]
            records.append(
                dict(
                    gallery_id=g["gallery_id"],
                    level=level,
                    temperature=temperature,
                    effective_positives_mean=float(entropy.exp().mean()),
                    diagonal_mass_mean=float(q.diagonal(dim1=-2, dim2=-1).mean()),
                    offdiagonal_similarity_mean=float(offdiag.mean()),
                    offdiagonal_similarity_max=float(offdiag.max()),
                    calibration_at_boundary=temperature in (float(candidates[0]), float(candidates[-1])),
                )
            )
    return temperatures, pd.DataFrame(records)


def window_logits(model: LinearHeads, gallery: dict, device: str, temperature: float) -> torch.Tensor:
    """Compute eight aligned query-by-candidate matrices with shared heads."""
    eeg = functional.normalize(model.eeg(gallery["eeg"].to(device)), dim=-1)
    movement = functional.normalize(model.movement(gallery["movement"].to(device)), dim=-1)
    return torch.einsum("ikd,jkd->kij", eeg, movement) / temperature


def objective(logits: torch.Tensor, gallery: dict, experiment: str, arm: str, alpha: float) -> torch.Tensor:
    """Contrast local windows or aggregate evidence, with coherent shuffled soft teachers."""
    level, kind = experiment.split("-")
    scores = logits if level == "local" else logits.mean(0)
    n = scores.shape[-1]
    mapping = gallery["mapping"].to(scores.device) if arm == "shuffled" else torch.arange(n, device=scores.device)
    scores = scores[..., mapping]
    target = torch.eye(n, device=scores.device)
    if kind == "soft":
        q = gallery[f"{level}_teacher"].to(scores.device)
        q = q[..., mapping, :][..., mapping]
        target = (1 - alpha) * target + alpha * q
    forward = -(target * scores.log_softmax(-1)).sum(-1).mean()
    reverse = -(target * scores.transpose(-2, -1).log_softmax(-1)).sum(-1).mean()
    return (forward + reverse) / 2


@torch.no_grad()
def evaluate(
    model: LinearHeads, sets: list[dict], device: str, temperature: float, experiment: str, arm: str, alpha: float
) -> pd.DataFrame:
    """Evaluate true trial/window ranks and condition-restricted candidate galleries."""
    model.eval()
    records = []
    for g in sets:
        logits = window_logits(model, g, device, temperature)
        assigned = objective(logits, g, experiment, arm if g["split"] == "train" else "true", alpha)
        matrices = [("trial", -1, logits.mean(0))] + [("window", k, logits[k]) for k in range(8)]
        for level, index, scores in matrices:
            for condition in ("all", "closed1", "closed2"):
                indices = [i for i, c in enumerate(g["conditions"]) if condition == "all" or c == condition]
                if not indices:
                    continue
                ix = torch.tensor(indices, device=device)
                for scope in ("mixed",) if condition == "all" else ("mixed", "within-condition"):
                    forward, reverse, positives = scores[ix], scores.T[ix], ix
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
                            level=level,
                            offset=OFFSETS[index] if index >= 0 else -1,
                            condition=condition,
                            scope=scope,
                            candidates=forward.shape[1],
                            queries=len(ix),
                            assigned_loss=float(assigned),
                            reverse_top1=backward["top1"],
                            reverse_mrr=backward["mrr"],
                            **metrics,
                        )
                    )
    return pd.DataFrame(records)


def fit(
    initial: LinearHeads,
    sets: list[dict],
    destination: Path,
    experiment: str,
    arm: str,
    schedule: list[list[int]],
    device: str,
    temperature: float,
    alpha: float,
    learning_rate: float,
    weight_decay: float,
    groups_per_update: int,
) -> dict:
    """Train one matched arm, selecting exclusively whole-trial validation MRR."""
    destination.mkdir(parents=True)
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
                    loss = objective(window_logits(model, g, device, temperature), g, experiment, arm, alpha)
                    require(bool(torch.isfinite(loss)), "Nonfinite loss")
                    (loss / len(chunk)).backward()
                require(
                    all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                    "Invalid gradients",
                )
                gradient_norm = max(gradient_norm, float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
                optimizer.step()
                updates += 1
        detail = evaluate(model, sets, device, temperature, experiment, arm, alpha)
        metrics = summarize(detail[(detail.level == "trial") & (detail.scope == "mixed")])
        window_metrics = summarize(detail[(detail.level == "window") & (detail.scope == "mixed")])
        row = dict(
            epoch=epoch,
            updates=updates,
            gradient_norm_max=gradient_norm,
            **metrics,
            **{f"window_{k}": v for k, v in window_metrics.items()},
        )
        history.append(row)
        pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
        plot_training_history(pd.DataFrame(history), destination, f"{arm}; whole-trial retrieval")
        print(f"{experiment}/{arm} epoch {epoch}: val MRR={metrics['validation_mrr']:.4f}", flush=True)
        if metrics["validation_mrr"] > best:
            best = metrics["validation_mrr"]
            selected = dict(experiment=experiment, arm=arm, best_epoch=epoch, **row)
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, destination / "best_heads.pt"
            )
            detail.to_csv(destination / "best_gallery_metrics.csv", index=False)
            scores = {}
            with torch.no_grad():
                for g in sets:
                    scores[g["gallery_id"]] = window_logits(model, g, device, temperature).mean(0).cpu().numpy()
            np.savez(destination / "best_scores.npz", **scores)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, destination / "last_heads.pt")
    detail.to_csv(destination / "last_gallery_metrics.csv", index=False)
    (destination / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def run_whole_trial(
    features: Path,
    run: Path,
    output: Path,
    epochs: int = 20,
    seed: int = 42,
    device: str = "cpu",
    learning_rate: float = 0.001,
    weight_decay: float = 0.1,
    temperature: float = 0.1,
    alpha: float = 0.5,
    effective_positives: float = 2.0,
    groups_per_update: int = 4,
) -> dict:
    """Run the four prespecified objectives with matched true/shuffled trial controls."""
    require(epochs > 0 and seed >= 0 and groups_per_update > 0, "Invalid budget/seed")
    require(all(math.isfinite(v) for v in (learning_rate, weight_decay, temperature, alpha)), "Nonfinite settings")
    require(learning_rate > 0 and temperature > 0 and weight_decay >= 0 and 0 < alpha < 1, "Invalid settings")
    bank = FeatureBank(features, run)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        operation="whole-trial",
        format_version=1,
        features_source=str(features.resolve()),
        split_source=str(run.resolve()),
        features_sha256=digest(features / "features.json"),
        split_sha256=digest(run / "split.json"),
        epochs=epochs,
        seed=seed,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        alpha=alpha,
        effective_positives=effective_positives,
        groups_per_update=groups_per_update,
        offsets=list(OFFSETS),
        readout="temporal",
        embedding_dim=32,
        experiments=list(EXPERIMENTS),
        selection="whole-trial dyad-macro validation MRR; epoch zero; earliest tie",
        test_features_accessed=False,
        reserved_features_accessed=False,
        weighting="equal eight overlapping windows, not independent samples",
        teacher="fixed centered phase sin/cos; local windows or unique 900-sample full trajectory",
        calibration="separate local/trial temperatures; median training-row effective positives",
        scope="Exploratory known-person held-trial retrieval; no significance claim",
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sets, stats = build_trials(bank, seed + 1042)
    torch.save(stats, output / "normalization.pt")
    temperatures, diagnostics = calibrate_teacher(sets, effective_positives)
    diagnostics.to_csv(output / "teacher_diagnostics.csv", index=False)
    for g in sets:
        for level in ("local", "trial"):
            g[f"{level}_teacher"] = teacher_probabilities(g.pop(f"{level}_similarity"), temperatures[level])
    torch.save(
        {g["gallery_id"]: {k: g[k] for k in ("local_teacher", "trial_teacher")} for g in sets}, output / "teachers.pt"
    )
    manifest = [
        {k: v for k, v in g.items() if not isinstance(v, torch.Tensor)}
        | {"shuffled_positive_columns": g["mapping"].tolist()}
        for g in sets
    ]
    (output / "galleries.json").write_text(json.dumps(manifest, indent=2) + "\n")
    schedule = epoch_schedule(sum(g["split"] == "train" for g in sets), epochs, seed + 2042)
    (output / "schedule.json").write_text(json.dumps(schedule) + "\n")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        initial = LinearHeads(800, 400, 32)
    torch.save(initial.state_dict(), output / "initial_heads.pt")
    comparison = []
    for experiment in EXPERIMENTS:
        for arm in ("true", "shuffled"):
            result = fit(
                initial,
                sets,
                output / experiment / arm,
                experiment,
                arm,
                schedule,
                device,
                temperature,
                alpha,
                learning_rate,
                weight_decay,
                groups_per_update,
            )
            comparison.append(result)
            pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    result = dict(
        **config,
        complete=True,
        teacher_temperatures=temperatures,
        results=comparison,
        versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        input_array_sha256={n: bank.metadata["artifact_sha256"][n] for n in sorted(bank.verified)},
        code_sha256={
            n: digest(Path(__file__).with_name(n))
            for n in ("whole_trial.py", "training.py", "tiny_fit.py", "models.py", "contrastive.py", "plotting.py")
        },
        output_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    (output / "whole_trial.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
