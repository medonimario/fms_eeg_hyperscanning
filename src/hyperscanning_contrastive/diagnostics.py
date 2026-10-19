"""CPU engineering controls on synthetic vectors; no EEG or LaBraM inference."""

import json
from pathlib import Path

import torch
from torch import nn

from .contrastive import contrastive_loss, retrieval


def run_controls(output: Path, seed: int = 42) -> dict:
    """Check identical views, tiny-set learning, held-out shared latents, and independent signals.

    Args:
        output: New directory for a reproducible diagnostic report.
        seed: Random seed for synthetic signals and model initialization.

    Returns:
        Measurements and explicit engineering pass/fail gates.
    """
    output.mkdir(parents=True, exist_ok=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        latent = torch.randn(160, 16)
        first = latent @ torch.randn(16, 24) + 0.05 * torch.randn(160, 24)
        second = latent @ torch.randn(16, 24) + 0.05 * torch.randn(160, 24)
        model_a, model_b = nn.Linear(24, 16), nn.Linear(24, 16)
        optimizer = torch.optim.Adam(list(model_a.parameters()) + list(model_b.parameters()), lr=0.01)
        initial = contrastive_loss(model_a(first[:32]), model_b(second[:32])).item()
        for _ in range(300):
            optimizer.zero_grad()
            loss = contrastive_loss(model_a(first[:32]), model_b(second[:32]))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            tiny = retrieval(model_a(first[:32]), model_b(second[:32]))
            held_out = retrieval(model_a(first[32:]), model_b(second[32:]))
            identical = retrieval(latent[32:], latent[32:])
            independent = []
            for _ in range(100):
                independent.append(retrieval(torch.randn(128, 16), torch.randn(128, 16))["top1"])
            independent_mean = sum(independent) / len(independent)
            final = contrastive_loss(model_a(first[:32]), model_b(second[:32])).item()
        gates = {
            "identical_views": identical["top1"] == 1,
            "tiny_set_overfit": tiny["top1"] >= 0.99 and final < initial * 0.1,
            "held_out_shared_latents": held_out["top1"] >= 0.8,
            "independent_signals": abs(independent_mean - 1 / 128) < 0.004,
        }
        report = {
            "scope": "Synthetic-vector engineering controls, not EEG/backbone validation or scientific inference",
            "seed": seed,
            "torch_version": torch.__version__,
            "training_pairs": 32,
            "held_out_pairs": 128,
            "steps": 300,
            "initial_loss": initial,
            "final_loss": final,
            "identical": identical,
            "tiny_set": tiny,
            "held_out_shared_latents": held_out,
            "independent_replicates": 100,
            "independent_top1_mean": independent_mean,
            "chance_top1": 1 / 128,
            "gates": gates,
            "passed": all(gates.values()),
        }
    (output / "controls.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
