"""Validation-only correspondence randomization for Stage 2 engineering checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import sha256
from .overfit import ContrastiveLabram
from .ssl import augment, gallery_metrics, load_evaluation

METRICS = ["top1", "recall5", "mrr", "mean_rank"]


def shuffle_metrics(first: torch.Tensor, second: torch.Tensor, rng: np.random.Generator) -> dict[str, float]:
    """Randomize designated correspondence with a uniform permutation, retaining diagonal evaluation labels.

    Fixed points are allowed to preserve the ordinary random-assignment chance rate. One permutation defines
    correspondence in both directions. This is an engineering null, not a temporal/physiological surrogate.
    """
    permutation = torch.as_tensor(rng.permutation(len(second)), device=second.device)
    return gallery_metrics(first, second[permutation])


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate members, conditions, and dyads equally for each randomization replicate."""
    return (
        frame.groupby(["repeat", "group", "condition"])[METRICS]
        .mean()
        .groupby(["repeat", "group"])
        .mean()
        .groupby("repeat")
        .mean()
    )


def check_correspondence(
    cache_path: Path, run: Path, output: Path, device: str = "cpu", permutations: int = 200, seed: int = 1042
) -> dict:
    """Compare real and shuffled validation matches for saved initial pooled, initial, and best features."""
    if permutations < 20:
        raise ValueError("Use at least 20 permutations for the engineering check.")
    cache, config, galleries = load_evaluation(cache_path, run, "validation")
    output.mkdir(parents=True, exist_ok=False)
    model = ContrastiveLabram("random", Path("unused"), False).to(device)
    model.eval()
    report = {
        "scope": "validation-only engineering control; not a hyperscanning significance test",
        "run": str(run.resolve()),
        "training_report_sha256": sha256(run / "ssl.json"),
        "cache_sha256": sha256(cache_path / "cache.json"),
        "permutations": permutations,
        "permutation_seed": seed,
        "device": device,
        "torch_version": torch.__version__,
        "test_evaluated": False,
        "fixed_points_allowed": True,
        "chance_top1": 1 / config.pairs,
        "chance_mrr": sum(1 / i for i in range(1, config.pairs + 1)) / config.pairs,
        "representations": {},
    }
    for name in ("initial_pooled", "initial", "best"):
        checkpoint = "initial" if name == "initial_pooled" else name
        model.load_state_dict(torch.load(run / f"{checkpoint}.pt", map_location="cpu", weights_only=True), strict=True)
        generator = torch.Generator().manual_seed(config.seed + 2000)
        rng = np.random.default_rng(seed)
        actual, null = [], []
        with torch.no_grad():
            for frame in galleries:
                signal = cache.signals(frame)
                a_view, b_view = (
                    augment(signal, config, generator).to(device),
                    augment(signal, config, generator).to(device),
                )
                encode = model.features if name == "initial_pooled" else model
                a, b = encode(a_view).cpu(), encode(b_view).cpu()
                identifiers = {
                    "group": int(frame.group.iloc[0]),
                    "condition": frame.condition.iloc[0],
                    "member": int(frame.member.iloc[0]),
                }
                actual.append({**identifiers, "repeat": 0, **gallery_metrics(a, b)})
                for repeat in range(permutations):
                    null.append({**identifiers, "repeat": repeat, **shuffle_metrics(a, b, rng)})
        actual_frame, null_frame = pd.DataFrame(actual), pd.DataFrame(null)
        actual_frame.to_csv(output / f"{name}_actual.csv", index=False)
        null_frame.to_csv(output / f"{name}_shuffled_galleries.csv", index=False)
        null_macro = summarize(null_frame)
        null_macro.to_csv(output / f"{name}_shuffled_macro.csv")
        report["representations"][name] = {
            "actual": summarize(actual_frame).iloc[0].to_dict(),
            "shuffled_mean": null_macro.mean().to_dict(),
            "shuffled_repeat_sd": null_macro.std().to_dict(),
            "shuffled_mean_mc_se": (null_macro.std() / np.sqrt(permutations)).to_dict(),
        }
        print(
            f"{name}: true top1={report['representations'][name]['actual']['top1']:.4f}, "
            f"shuffled top1={null_macro.top1.mean():.4f}",
            flush=True,
        )
    (output / "correspondence.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
