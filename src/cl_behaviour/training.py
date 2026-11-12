"""Within-person held-trial retrieval with shared heads and fixed, matched controls."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .contrastive import paired_logits, retrieval_metrics, symmetric_loss
from .models import LinearHeads
from .tiny_fit import READOUTS, center_scale, checked_path, digest, readout, require

OFFSETS = (0, 4, 8, 12)


class FeatureBank:
    """Validate prepared feature/split identities before loading permitted feature arrays."""

    def __init__(self, features: Path, run: Path) -> None:
        """Require complete combined-closed training/validation coverage and original galleries."""
        self.features = features
        self.metadata = json.loads((features / "features.json").read_text())
        self.protocol = json.loads((run / "split.json").read_text())
        expected = dict(
            operation="features",
            format_version=1,
            complete=True,
            profile="combined-closed",
            members=["a", "b"],
            token_shape=[2, 64, 4, 200],
            token_order="channel-major, time-minor",
            token_dtype="float32",
            movement_shape=[2, 2, 900],
            movement_components=["sin", "cos"],
            movement_sfreq=50,
            centered=False,
            statistics_fitted=False,
            backbone_frozen=True,
        )
        for key, value in expected.items():
            require(self.metadata.get(key) == value, f"Invalid feature contract: {key}")
        require(self.protocol.get("operation") == "split" and self.protocol.get("format_version") == 2, "Invalid split")
        require(self.protocol["profile"] == "combined-closed", "Expected combined-closed split")
        require(digest(run / "split.json") == self.metadata["split_sha256"], "Split hash mismatch")
        require(self.protocol["cache_sha256"] == self.metadata["cache_sha256"], "Preparation mismatch")
        originals, frames = {}, {}
        for name in ("trials", "windows"):
            original_name = "splits.csv" if name == "trials" else "windows.csv"
            originals[name] = pd.read_csv(
                checked_path(run, original_name, self.protocol["manifest_sha256"]), keep_default_na=False
            )
            frames[name] = pd.read_csv(
                checked_path(features, f"{name}.csv", self.metadata["artifact_sha256"]), keep_default_na=False
            )
            key = "trial_id" if name == "trials" else "pair_id"
            original, actual = originals[name], frames[name]
            allowed = original[original.eligible & original.split.isin(["train", "validation"])]
            require(not actual[key].duplicated().any(), f"Duplicate {key}")
            require(set(actual[key]) == set(allowed[key]), f"Incomplete or forbidden {name} coverage")
            aligned = allowed.set_index(key, drop=False).loc[actual[key]].reset_index(drop=True)
            for column in original.columns:
                require((actual[column].reset_index(drop=True) == aligned[column]).all(), f"Mismatch: {name}/{column}")
            require(not actual.cohort_fold.isin([1, 2]).any(), "Reserved dyads in features")
        self.trials, self.windows = frames["trials"], frames["windows"]
        require(set(self.windows.trial_id) == set(self.trials.trial_id), "Feature parent coverage mismatch")
        for _, rows in self.windows.groupby("trial_id"):
            require(sorted(rows.crop_start_s) == list(range(0, 15, 2)), "Incomplete offsets")
        parent = self.trials.set_index("trial_id").loc[self.windows.trial_id]
        require(
            np.array_equal(parent.movement_feature_index, self.windows.movement_feature_index), "Movement row mismatch"
        )
        require(
            np.array_equal(parent.movement_feature_array, self.windows.movement_feature_array), "Movement file mismatch"
        )
        require(self.windows.movement_feature_start.eq(self.windows.crop_start_s * 50).all(), "Movement start mismatch")
        require(
            self.windows.movement_feature_stop.eq((self.windows.crop_start_s + 4) * 50).all(), "Movement stop mismatch"
        )
        galleries = pd.read_csv(checked_path(run, "galleries.csv", self.protocol["manifest_sha256"]))
        self.galleries = galleries[galleries.split.isin(["train", "validation"])].copy()
        require(not self.galleries.gallery_id.duplicated().any(), "Duplicate galleries")
        actual_keys = set(zip(self.galleries.group, self.galleries.member, self.galleries.split, strict=True))
        expected_keys = {(g, m, s) for g in self.trials.group.unique() for m in "ab" for s in ("train", "validation")}
        require(actual_keys == expected_keys and len(self.galleries) == len(expected_keys), "Gallery coverage mismatch")
        for row in self.galleries.itertuples():
            expected_trials = self.trials[(self.trials.group == row.group) & (self.trials.split == row.split)]
            expected_trials = expected_trials.sort_values(["condition_trial", "condition"])
            require(json.loads(row.trial_ids) == expected_trials.trial_id.tolist(), "Gallery trial order mismatch")
            require(
                json.loads(row.trial_conditions) == expected_trials.condition.tolist(), "Gallery conditions mismatch"
            )
            require(row.candidate_count == len(expected_trials) >= 2, "Invalid gallery size")
            require(row.main_summary_eligible == (row.group != 2020), "Summary eligibility mismatch")
        norm_path = checked_path(features, "pooling_norm.npz", self.metadata["artifact_sha256"])
        with np.load(norm_path, allow_pickle=False) as norm:
            self.norm = {key: np.array(norm[key], copy=True) for key in ("weight", "bias", "eps")}
        require(self.norm["weight"].shape == self.norm["bias"].shape == (200,), "Invalid normalization shape")
        require(all(np.isfinite(v).all() for v in self.norm.values()) and float(self.norm["eps"]) > 0, "Invalid norm")
        self.verified: set[str] = set()

    def array(self, name: str, shape: tuple) -> np.ndarray:
        """Verify each permitted source once and map its exact expected shape."""
        if name not in self.verified:
            checked_path(self.features, name, self.metadata["artifact_sha256"])
            self.verified.add(name)
        array = np.load(self.features / name, mmap_mode="r", allow_pickle=False)
        require(array.shape == shape and array.dtype == np.float32, f"Invalid feature array: {name}")
        return array

    def build(self, profile: str, permutation_seed: int) -> tuple[list[dict], dict]:
        """Build readout galleries and fit participant-specific statistics on training windows only."""
        sets, statistics = [], {}
        for group, trial_rows in self.trials.groupby("group", sort=True):
            print(f"Loading {profile} features for pair {group}", flush=True)
            window_rows = self.windows[self.windows.group == group]
            require(window_rows.token_array.eq(f"{group}_tokens.npy").all(), "Token filename mismatch")
            require(trial_rows.movement_feature_array.eq(f"{group}_movement.npy").all(), "Movement filename mismatch")
            for rows, column in ((window_rows, "token_array_index"), (trial_rows, "movement_feature_index")):
                require(sorted(rows[column]) == list(range(len(rows))), f"Invalid {column}")
            tokens = self.array(f"{group}_tokens.npy", (len(window_rows), 2, 64, 4, 200))
            movement = self.array(f"{group}_movement.npy", (len(trial_rows), 2, 2, 900))
            for member_index, member in enumerate("ab"):
                person_sets = []
                for gallery in self.galleries[
                    (self.galleries.group == group) & (self.galleries.member == member)
                ].itertuples():
                    ids = json.loads(gallery.trial_ids)
                    for offset in OFFSETS:
                        rows = window_rows[window_rows.crop_start_s == offset].set_index("trial_id").loc[ids]
                        raw = torch.from_numpy(
                            np.array(tokens[rows.token_array_index.to_numpy(), member_index], copy=True)
                        )
                        behaviour = torch.from_numpy(
                            np.array(
                                movement[
                                    rows.movement_feature_index.to_numpy(),
                                    member_index,
                                    :,
                                    offset * 50 : (offset + 4) * 50,
                                ],
                                copy=True,
                            )
                        ).flatten(1)
                        require(
                            bool(torch.isfinite(raw).all() and torch.isfinite(behaviour).all()), "Nonfinite features"
                        )
                        person_sets.append(
                            dict(
                                gallery_id=f"{gallery.gallery_id}:{offset}",
                                group=int(group),
                                member=member,
                                split=gallery.split,
                                offset=offset,
                                trial_ids=ids,
                                conditions=json.loads(gallery.trial_conditions),
                                eeg=readout(raw, self.norm, profile),
                                movement=behaviour,
                            )
                        )
                train_sets = [g for g in person_sets if g["split"] == "train"]
                stats = {}
                for modality in ("eeg", "movement"):
                    _, fitted = center_scale(torch.cat([g[modality] for g in train_sets]))
                    stats[modality] = fitted
                    for gallery in person_sets:
                        gallery[modality] = (gallery[modality] - fitted["mean"]) / fitted["scale"]
                statistics[f"{group}:{member}"] = stats
                rng = torch.Generator().manual_seed(permutation_seed + int(group) * 2 + member_index)
                mapping = torch.randperm(len(train_sets[0]["trial_ids"]), generator=rng)
                for gallery in person_sets:
                    gallery["mapping"] = (
                        mapping if gallery["split"] == "train" else torch.arange(len(gallery["trial_ids"]))
                    )
                sets.extend(person_sets)
            del tokens, movement
        return sets, statistics


@torch.no_grad()
def evaluate(model: LinearHeads, sets: list[dict], device: str, temperature: float, arm: str) -> pd.DataFrame:
    """Score fixed galleries against true correspondence, plus assigned training loss."""
    model.eval()
    records = []
    for gallery in sets:
        logits = paired_logits(
            model.eeg(gallery["eeg"].to(device)), model.movement(gallery["movement"].to(device)), temperature
        )
        mapping = gallery["mapping"].to(device) if arm == "shuffled" and gallery["split"] == "train" else None
        assigned_loss = float(symmetric_loss(logits, mapping))
        for condition in ("all", "closed1", "closed2"):
            indices = [i for i, c in enumerate(gallery["conditions"]) if condition == "all" or c == condition]
            if not indices:
                continue
            positives = torch.tensor(indices, device=device, dtype=torch.long)
            metrics = retrieval_metrics(logits[positives], positives)
            reverse = retrieval_metrics(logits.T[positives], positives)
            records.append(
                dict(
                    gallery_id=gallery["gallery_id"],
                    group=gallery["group"],
                    member=gallery["member"],
                    split=gallery["split"],
                    offset=gallery["offset"],
                    condition=condition,
                    queries=len(indices),
                    candidates=len(gallery["trial_ids"]),
                    assigned_loss=assigned_loss,
                    reverse_top1=reverse["top1"],
                    reverse_mrr=reverse["mrr"],
                    **metrics,
                )
            )
    return pd.DataFrame(records)


def summarize(frame: pd.DataFrame) -> dict:
    """Average offsets/members within dyads, then equally weight eligible dyads."""
    result = {}
    for split in ("train", "validation"):
        rows = frame[(frame.split == split) & (frame.condition == "all") & (frame.group != 2020)]
        require(not rows.empty, "No main-summary dyads")
        for metric in ("top1", "mrr", "chance_top1", "chance_mrr", "assigned_loss", "reverse_top1", "reverse_mrr"):
            result[f"{split}_{metric}"] = float(rows.groupby("group")[metric].mean().mean())
    return result


def epoch_schedule(count: int, epochs: int, seed: int) -> list[list[int]]:
    """Save a deterministic exhaustive gallery order shared across arms and readouts."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(count).tolist() for _ in range(epochs)]


def fit_arm(
    initial: LinearHeads,
    sets: list[dict],
    output: Path,
    arm: str,
    schedule: list[list[int]],
    device: str,
    temperature: float,
    learning_rate: float,
    weight_decay: float,
    groups_per_update: int,
) -> tuple[dict, list[dict]]:
    """Fit shared heads; select strictly improving validation MRR, including epoch zero."""
    from .plotting import plot_training_history

    output.mkdir(parents=True, exist_ok=False)
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_sets = [g for g in sets if g["split"] == "train"]
    history, best, best_epoch, updates = [], -math.inf, 0, 0
    for epoch in range(len(schedule) + 1):
        model.train()
        gradient_norm = 0.0
        if epoch:
            order = schedule[epoch - 1]
            for start in range(0, len(order), groups_per_update):
                chunk = order[start : start + groups_per_update]
                optimizer.zero_grad()
                for index in chunk:
                    gallery = train_sets[index]
                    logits = paired_logits(
                        model.eeg(gallery["eeg"].to(device)),
                        model.movement(gallery["movement"].to(device)),
                        temperature,
                    )
                    mapping = gallery["mapping"].to(device) if arm == "shuffled" else None
                    loss = symmetric_loss(logits, mapping) / len(chunk)
                    require(bool(torch.isfinite(loss)), "Nonfinite loss")
                    loss.backward()
                require(
                    all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                    "Invalid gradients",
                )
                gradient_norm = max(gradient_norm, float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
                optimizer.step()
                updates += 1
        detail = evaluate(model, sets, device, temperature, arm)
        metrics = summarize(detail)
        history.append(dict(epoch=epoch, updates=updates, gradient_norm_max=gradient_norm, **metrics))
        temporary = output / ".history.csv"
        pd.DataFrame(history).to_csv(temporary, index=False)
        temporary.replace(output / "history.csv")
        plot_training_history(pd.DataFrame(history), output, arm)
        print(
            f"{output.parent.name}/{arm} epoch {epoch}: validation top1={metrics['validation_top1']:.3f}, "
            f"MRR={metrics['validation_mrr']:.3f}",
            flush=True,
        )
        if metrics["validation_mrr"] > best:
            best, best_epoch = metrics["validation_mrr"], epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(state, output / "best_heads.pt")
            detail.to_csv(output / "best_gallery_metrics.csv", index=False)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, output / "last_heads.pt")
    detail.to_csv(output / "last_gallery_metrics.csv", index=False)
    selected = history[best_epoch]
    summary = dict(arm=arm, best_epoch=best_epoch, **{k: v for k, v in selected.items() if k != "epoch"})
    (output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, history


def run_training(
    features: Path,
    run: Path,
    output: Path,
    profiles: tuple[str, ...] = READOUTS,
    epochs: int = 20,
    device: str = "cpu",
    seed: int = 42,
    groups_per_update: int = 4,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    temperature: float = 0.1,
) -> dict:
    """Run validation-selected shared-head baselines with untouched within-dyad and external tests."""
    require(
        bool(profiles) and len(set(profiles)) == len(profiles) and set(profiles) <= set(READOUTS), "Invalid readouts"
    )
    require(epochs > 0 and groups_per_update > 0 and seed >= 0, "Invalid training budget/seed")
    require(all(math.isfinite(v) for v in (learning_rate, weight_decay, temperature)), "Nonfinite settings")
    require(learning_rate > 0 and temperature > 0 and weight_decay >= 0, "Invalid optimization settings")
    bank = FeatureBank(features, run)
    output.mkdir(parents=True, exist_ok=False)
    comparison = []
    config = dict(
        format_version=1,
        operation="train",
        features_source=str(features.resolve()),
        split_source=str(run.resolve()),
        features_sha256=digest(features / "features.json"),
        split_sha256=digest(run / "split.json"),
        profiles=list(profiles),
        epochs=epochs,
        device=device,
        seed=seed,
        offsets=list(OFFSETS),
        groups_per_update=groups_per_update,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        embedding_dim=32,
        profile="combined-closed",
        target="self",
        initialization="fresh heads; frozen cached LaBraM; no tiny-fit weights",
        normalization="participant training-only mean and scalar RMS vector norm",
        selection="dyad-macro validation EEG-to-movement MRR; epoch zero eligible; strict improvement, earliest tie",
        main_summary="equal dyads after member/offset means; 2020 excluded from summaries, included in fitting",
        training_galleries=len(bank.galleries[bank.galleries.split == "train"]) * len(OFFSETS),
        validation_galleries=len(bank.galleries[bank.galleries.split == "validation"]) * len(OFFSETS),
        candidate_ranges=bank.galleries.groupby("split").candidate_count.agg(["min", "max"]).to_dict("index"),
        train_participant_windows=int(bank.galleries[bank.galleries.split == "train"].candidate_count.sum())
        * len(OFFSETS),
        validation_participant_windows=int(bank.galleries[bank.galleries.split == "validation"].candidate_count.sum())
        * len(OFFSETS),
        test_features_accessed=False,
        reserved_features_accessed=False,
        scope="Exploratory within-person held-trial window retrieval; no significance claim",
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for profile in profiles:
        destination = output / profile
        destination.mkdir()
        sets, stats = bank.build(profile, seed + 1042)
        torch.save(stats, destination / "normalization.pt")
        training_sets = [g for g in sets if g["split"] == "train"]
        schedule = epoch_schedule(len(training_sets), epochs, seed + 2042)
        manifest = [
            {key: value for key, value in g.items() if key not in ("eeg", "movement", "mapping")}
            | {"shuffled_positive_columns": g["mapping"].tolist()}
            for g in sets
        ]
        (destination / "galleries.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (destination / "schedule.json").write_text(json.dumps(schedule) + "\n")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            initial = LinearHeads(sets[0]["eeg"].shape[1], 400, 32)
        for arm in ("true", "shuffled"):
            summary, _ = fit_arm(
                initial,
                sets,
                destination / arm,
                arm,
                schedule,
                device,
                temperature,
                learning_rate,
                weight_decay,
                groups_per_update,
            )
            comparison.append(dict(profile=profile, **summary))
            pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
        del sets, stats, training_sets
    selected = max((row for row in comparison if row["arm"] == "true"), key=lambda row: row["validation_mrr"])
    result = dict(
        **config,
        complete=True,
        selected_true_readout=selected["profile"],
        results=comparison,
        input_array_sha256={name: bank.metadata["artifact_sha256"][name] for name in sorted(bank.verified)},
        versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        code_sha256={
            name: digest(Path(__file__).with_name(name))
            for name in ("training.py", "tiny_fit.py", "contrastive.py", "models.py", "plotting.py")
        },
        output_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    (output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
