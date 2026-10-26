"""Training-only centering and compact structured frozen-LaBraM heads."""

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

from .contrastive import contrastive_loss
from .data import sha256
from .overfit import ContrastiveLabram
from .partner_patches import TokenExtractor, reduce_tokens
from .video_cnn import batch, evaluate, load_bank, make_schedule, plot_history
from .video_labram import load_reference


class VideoHead(nn.Module):
    """Small nonlinear pooled head or token-shared temporal head with ordered concatenation."""

    def __init__(self, profile: str) -> None:
        """Keep head capacities comparable without flattening all spatial tokens."""
        super().__init__()
        if profile not in ("pooled", "temporal", "spatial", "full"):
            raise ValueError("Unsupported profile.")
        self.profile = profile
        width = {"pooled": 64, "temporal": 32, "spatial": 48, "full": 16}[profile]
        self.token = nn.Linear(200, width)
        if profile in ("spatial", "full"):
            regions = 2 if profile == "spatial" else 3
            self.channel_mix = nn.Linear(64, regions, bias=False)
            inputs = regions * width * (4 if profile == "full" else 1)
        else:
            inputs = width if profile == "pooled" else 4 * width
        self.output = nn.Linear(inputs, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map centered features to a shared retrieval embedding."""
        expected = {"pooled": (200,), "temporal": (4, 200), "spatial": (64, 200), "full": (256, 200)}
        if tuple(x.shape[1:]) != expected[self.profile]:
            raise ValueError(f"Unexpected {self.profile} feature shape: {tuple(x.shape)}")
        z = torch.nn.functional.gelu(self.token(x))
        if self.profile == "spatial":
            z = self.channel_mix(z.transpose(1, 2)).transpose(1, 2)
        elif self.profile == "full":
            z = z.reshape(-1, 64, 4, 16).permute(0, 2, 3, 1)
            z = self.channel_mix(z).permute(0, 3, 1, 2)
        return self.output(z.flatten(1))


def center_bank(bank: dict, mode: str) -> tuple[dict, dict, list]:
    """Fit global or calibrated per-person feature means using training intervals only."""
    train = bank["train"]
    people = sorted({p for p, _ in train})
    moments = {p: {t for person, t in train if person == p} for p in people}
    common = sorted(set.intersection(*(moments[p] for p in people)))
    if mode == "none":
        centers = {"global": np.zeros_like(next(iter(train.values())))}
    elif mode == "global":
        centers = {"global": np.mean(np.stack(list(train.values())), axis=0)}
    elif mode == "participant":
        if len(common) < 4:
            raise ValueError("Participant calibration requires at least four training moments shared by every viewer.")
        centers = {p: np.mean(np.stack([train[(p, t)] for t in common]), axis=0) for p in people}
    else:
        raise ValueError("Unknown centering mode.")
    result = {
        s: {key: x - centers[key[0] if mode == "participant" else "global"] for key, x in values.items()}
        for s, values in bank.items()
    }
    return result, centers, common


def geometry(bank: dict) -> dict:
    """Compare matched-time cross-person and different-time within-person cosine descriptively."""
    output = {}
    for split, values in bank.items():
        normalized = {k: x.reshape(-1) / max(np.linalg.norm(x), 1e-12) for k, x in values.items()}
        people = sorted({p for p, _ in values})
        times = sorted({t for _, t in values})
        within, aligned = [], []
        for p in people:
            vectors = [normalized[(p, t)] for t in times if (p, t) in normalized]
            within.extend(float(a @ b) for a, b in combinations(vectors, 2))
        for t in times:
            vectors = [normalized[(p, t)] for p in people if (p, t) in normalized]
            aligned.extend(float(a @ b) for a, b in combinations(vectors, 2))
        output[split] = dict(
            same_person_different_time=float(np.mean(within)), different_person_same_time=float(np.mean(aligned))
        )
    return output


def extract_banks(
    raw: dict,
    checkpoint: Path,
    device: str,
    batch_size: int,
    output: Path,
    profiles: tuple[str, ...] = ("pooled", "temporal"),
) -> tuple[dict, float]:
    """Extract both readouts in one frozen pass with original LaBraM signal scaling."""
    model = ContrastiveLabram("pretrained", checkpoint, False).to(device).eval()
    model.requires_grad_(False)
    extractor = TokenExtractor(model).eval()
    banks = {p: {s: {} for s in raw} for p in profiles}
    indices = []
    with torch.inference_mode():
        for split, values in raw.items():
            keys = sorted(values)
            for start in range(0, len(keys), batch_size):
                selected = keys[start : start + batch_size]
                x = torch.from_numpy(np.stack([values[k] for k in selected])).to(device)
                tokens = extractor.features(x)
                for profile in banks:
                    z = reduce_tokens(tokens, profile, model.backbone.fc_norm).cpu().numpy()
                    banks[profile][split].update(zip(selected, z, strict=True))
                indices.extend(dict(split=split, participant_id=p, cache_start=t) for p, t in selected)
                print(f"Extracted {split}: {min(start + batch_size, len(keys))}/{len(keys)}", flush=True)
    pd.DataFrame(indices).to_csv(output / "feature_windows.csv", index=False)
    for profile in banks:
        np.save(
            output / f"{profile}_features.npy",
            np.stack([banks[profile][r["split"]][(r["participant_id"], r["cache_start"])] for r in indices]),
        )
    return banks, extractor.max_reconstruction_error


def run_video_projection(
    cache: Path,
    run: Path,
    output: Path,
    checkpoint: Path,
    device: str = "cpu",
    epochs: int = 20,
    steps_per_epoch: int = 100,
    pairs: int = 8,
    temperature: float = 0.5,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    seed: int = 42,
    batch_size: int = 16,
    centerings: tuple[str, ...] = ("global", "participant"),
    profiles: tuple[str, ...] = ("pooled", "temporal"),
    diagnostics_only: bool = False,
) -> dict:
    """Run fixed geometry ablations followed by matched frozen-feature contrastive heads."""
    if min(epochs, steps_per_epoch, batch_size) < 1 or pairs < 2:
        raise ValueError("Invalid run sizes.")
    if (
        not np.isfinite([temperature, learning_rate, weight_decay]).all()
        or min(temperature, learning_rate) <= 0
        or weight_decay < 0
    ):
        raise ValueError("Invalid optimization settings.")
    if not centerings or len(set(centerings)) != len(centerings) or set(centerings) - {"none", "global", "participant"}:
        raise ValueError("Invalid centering profiles.")
    if not profiles or len(set(profiles)) != len(profiles) or set(profiles) - {"pooled", "temporal", "spatial", "full"}:
        raise ValueError("Invalid readout profiles.")
    cache, run = cache.resolve(), run.resolve()
    reference, galleries, windows = load_reference(cache, run)
    raw, _ = load_bank(cache, run, windows, standardize=False)
    schedule, census = make_schedule(raw["train"], pairs, epochs * steps_per_epoch, seed)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("galleries.csv", "participant_splits.csv", "window_splits.csv"):
        shutil.copyfile(run / name, output / name)
    census.to_csv(output / "training_pair_census.csv", index=False)
    pd.DataFrame(
        [dict(step=i, first=a, second=b, starts=json.dumps(t), rotation=r) for i, (a, b, t, r) in enumerate(schedule)]
    ).to_csv(output / "schedule.csv", index=False)
    banks, reconstruction = extract_banks(raw, checkpoint, device, batch_size, output, profiles)
    del raw
    diagnostics, summaries, comparison = {}, {}, []
    for profile, original in banks.items():
        diagnostics[profile] = {}
        for mode in ("none", "global", "participant"):
            centered, centers, common = center_bank(original, mode)
            label = f"{profile}_{mode}"
            folder = output / label
            folder.mkdir()
            np.savez(folder / "centers.npz", **centers)
            scores = {}
            for split in ("train", "validation"):
                scores[split], people = evaluate(
                    nn.Flatten(start_dim=1), centered[split], galleries[galleries.split == split], device, temperature
                )
                people.to_csv(folder / f"direct_{split}_people.csv")
                comparison.append(dict(profile=profile, centering=mode, arm="direct", split=split, **scores[split]))
            diagnostics[profile][mode] = dict(
                retrieval=scores,
                geometry=geometry(centered),
                calibration_moments=common if mode == "participant" else [],
                center_fit="matched training moments per person" if mode == "participant" else "all training windows",
            )
            if diagnostics_only or mode not in centerings:
                continue
            torch.manual_seed(seed)
            initial = VideoHead(profile).to(device)
            torch.save(initial.state_dict(), folder / "initial.pt")
            histories, arms = {}, {}
            for arm in ("true", "misaligned"):
                destination = folder / arm
                destination.mkdir()
                model = copy.deepcopy(initial)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                history, best, best_epoch, coverage = [], -np.inf, 0, set()
                for epoch in range(epochs + 1):
                    losses, used = [], set()
                    if epoch:
                        model.train()
                        for entry in schedule[(epoch - 1) * steps_per_epoch : epoch * steps_per_epoch]:
                            a, b = batch(centered["train"], entry, device, arm == "misaligned")
                            loss = contrastive_loss(model(a), model(b), temperature)
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite loss.")
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                            optimizer.step()
                            losses.append(float(loss.detach()))
                            used.update((p, t) for p in entry[:2] for t in entry[2])
                        coverage.update(used)
                    row = dict(
                        epoch=epoch,
                        objective_loss=float(np.mean(losses)) if losses else None,
                        epoch_unique_windows=len(used),
                        cumulative_unique_windows=len(coverage),
                    )
                    for split in ("train", "validation"):
                        metrics, people = evaluate(
                            model, centered[split], galleries[galleries.split == split], device, temperature
                        )
                        row.update({f"{split}_{k}": v for k, v in metrics.items()})
                        if split == "validation" and metrics["mrr"] > best:
                            best, best_epoch = metrics["mrr"], epoch
                            torch.save(model.state_dict(), destination / "best.pt")
                            people.to_csv(destination / "best_validation_people.csv")
                    history.append(row)
                    pd.DataFrame(history).to_csv(destination / "history.csv", index=False)
                    print(f"{label}/{arm} epoch {epoch}: validation MRR={row['validation_mrr']:.4f}", flush=True)
                torch.save(model.state_dict(), destination / "last.pt")
                histories[arm] = pd.DataFrame(history)
                arms[arm] = dict(best_epoch=best_epoch, selected=history[best_epoch], final=history[-1])
                for split in ("train", "validation"):
                    comparison.append(
                        dict(
                            profile=profile,
                            centering=mode,
                            arm=arm,
                            split=split,
                            **{m: history[best_epoch][f"{split}_{m}"] for m in ("top1", "mrr", "loss")},
                        )
                    )
            plot_history(histories, folder, reference["metrics"])
            summaries[label] = dict(parameters=sum(p.numel() for p in initial.parameters()), arms=arms)
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    plot_summary(pd.DataFrame(comparison), output)
    report = dict(
        scope="Frozen pretrained LaBraM geometry and contrastive readouts; exploratory held-time pilot",
        cache_sha256=sha256(cache / "cache.json"),
        reference_sha256=sha256(run / "baseline.json"),
        checkpoint_sha256=sha256(checkpoint),
        gallery_sha256=sha256(output / "galleries.csv"),
        reconstruction_max_error=reconstruction,
        seed=seed,
        device=device,
        torch_version=torch.__version__,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pairs=pairs,
        temperature=temperature,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        centerings=centerings,
        profiles=profiles,
        diagnostics_only=diagnostics_only,
        readouts={
            p: {
                "shape": list(next(iter(banks[p]["train"].values())).shape),
                "head_parameters": sum(x.numel() for x in VideoHead(p).parameters()),
            }
            for p in profiles
        },
        direct_similarity="Flatten corresponding token positions then global L2 normalization; no cross-channel search",
        structured_head="Shared token compression then signed learned channel mixtures; full preserves four time bins",
        training_person_windows=len(banks[profiles[0]]["train"]),
        validation_person_windows=len(banks[profiles[0]]["validation"]),
        reserved_people_evaluated=False,
        backbone_frozen=True,
        input="original microvolts divided by100 once",
        temporal_readout="channel average then frozen fc_norm per time position; four ordered tokens",
        calibration="Participant centering needs own training intervals; unseen-person calibration is not evaluated",
        similarity="All correlation-named metrics in this run are embedding cosine similarities",
        selection="validation MRR including epoch zero, earliest tie; variants not independently tested",
        diagnostics=diagnostics,
        experiments=summaries,
        artifacts_sha256={str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()},
    )
    (output / "projection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def plot_summary(frame: pd.DataFrame, output: Path) -> None:
    """Plot direct and selected-head validation scores without implying independent tests."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = frame[frame.split == "validation"].copy()
    frame["label"] = frame.profile + "/" + frame.centering
    labels = list(dict.fromkeys(frame.label))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, chance in zip(axes, ("top1", "mrr"), (0.25, sum(1 / np.arange(1, 5)) / 4), strict=True):
        for arm, rows in frame.groupby("arm"):
            ax.plot(labels, rows.set_index("label")[metric].reindex(labels), "o-", label=arm)
        ax.axhline(chance, color="gray", linestyle=":")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
    fig.suptitle("Validation: direct readouts and validation-selected heads (exploratory)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"comparison.{extension}", dpi=160)
    plt.close(fig)
