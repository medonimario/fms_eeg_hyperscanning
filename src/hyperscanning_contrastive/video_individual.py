"""Independent per-movie baselines and frozen LaBraM readout training."""

from pathlib import Path

from .video_baseline import run_video_baseline
from .video_projection import run_video_projection


def run_individual_video(
    cache: Path,
    reference: Path,
    output: Path,
    checkpoint: Path,
    video_id: int,
    noise_level: str = "combined",
    device: str = "cpu",
    epochs: int = 20,
    steps_per_epoch: int = 100,
    pairs: int = 8,
    temperature: float = 0.5,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    seed: int = 42,
    batch_size: int = 16,
    profiles: tuple[str, ...] = ("spatial", "full"),
    centerings: tuple[str, ...] = ("global", "participant"),
    diagnostics_only: bool = False,
    head_type: str = "compact",
    cohort: str = "solo",
) -> dict:
    """Build a movie-specific reference and train fresh heads without sharing movie statistics."""
    if video_id not in (3, 4, 5, 6, 7, 8, 9, 10):
        raise ValueError("Choose another movie: 3,4,5,6,7,8,9,10; preserve existing video0 results.")
    output.mkdir(parents=True, exist_ok=False)
    run_video_baseline(
        cache,
        output / "baseline",
        seed=seed,
        video_id=video_id,
        noise_level=noise_level,
        reference=reference,
        cohort=cohort,
    )
    return run_video_projection(
        cache,
        output / "baseline",
        output / "projection",
        checkpoint,
        device,
        epochs,
        steps_per_epoch,
        pairs,
        temperature,
        learning_rate,
        weight_decay,
        seed,
        batch_size,
        centerings,
        profiles,
        diagnostics_only,
        head_type,
    )
