# LaBraM without pretraining

`--model labram --initialization random` constructs the same native Braindecode LaBraM architecture
as the pretrained path without reading, hashing, or downloading a checkpoint. It works with every
existing cache and uses the same triad/dyad partitions, input scaling (microvolts / 100), binary loss,
validation-based selection, W&B metrics, prediction exports, and mean ± SD plotting.

## What changes

The backbone retains Braindecode's native initialization rather than loading student weights.
Random initialization includes the usual deterministic initialization of biases and normalization
parameters; it does not mean sampling every tensor randomly. The 12 transformer blocks, embeddings,
mean pooling, disabled QKV bias and fresh single-output head are identical in architecture to the
pretrained path. Both paths retain the existing head initialization (truncated normal with std 0.02,
then multiplied by 0.001; zero bias). Every fold resets the configured random seed before construction.

The scratch starting profile is:

| Parameter | Random LaBraM | Pretrained LaBraM |
| --- | --- | --- |
| Epochs | 100 | 10 |
| Learning rate | 1e-4 for all datasets | MG 1e-4; FG/Circling 5e-4 |
| Layer decay | 1, uniform rates | 0.65 |
| Warmup | 5 epochs | 5 epochs |
| Batch size / accumulation | 128 / 2 | 128 / 2 |
| Weight decay | 0.05 | 0.05 |
| Stochastic depth | 0.2 | 0.2 |

Both use AdamW and the same cosine schedule. These scratch settings are a starting point, not tuned
results. More epochs give scratch training time to learn representations; uniform layer rates avoid
suppressing updates to randomly initialized early layers. Inspect validation convergence before
interpreting a poor scratch result as evidence that pretraining is necessary.

For a strict initialization-only comparison, explicitly match all optimizer/schedule settings between
pretrained and random runs. For example, the old MG protocol is `--epochs 10 --warmup-epochs 5
--lr 1e-4 --layer-decay 0.65 --batch-size 128 --update-freq 2 --weight-decay 0.05 --drop-path 0.2`.
That comparison estimates pretraining benefit under a fixed short budget. Comparing the default
100-epoch scratch profile against 10-epoch fine-tuning also changes the optimization protocol;
report both profiles rather than calling it an initialization-only ablation.

## Run

From the repository root in the existing environment:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m labram_classification cv --model labram --initialization random \
  --cache /work3/s204684/Thesis/data/processed/mg_solo_coordination \
  --output models/results/mg_labram_random_run1 --device cuda \
  --wandb-mode online --wandb-group mg_labram_random_run1
```

No checkpoint is required; a supplied `--checkpoint` is unused in random mode. Existing commands
default to pretrained LaBraM. EEGNet resolves to random initialization and rejects `pretrained`.

For an inexpensive wiring check before full training:

```bash
python -m labram_classification train --model labram --initialization random \
  --cache /path/to/cache --output runs/labram_random_smoke --fold 0 \
  --epochs 2 --warmup-epochs 0 --batch-size 2 --update-freq 1 \
  --smoke-windows-per-class 2 --num-workers 0 --device cpu --wandb-mode disabled
```

For HPC, export `PROJECT_DIR`, `VENV`, `CACHE_DIR`, a new `OUTPUT_DIR`, and the desired W&B
settings as with EEGNet, then submit `bsub < src/labram_classification/hpc_templates/labram_random_cv.lsf`.
Create `outputs/` and `errors/` before submission. The template accepts `EPOCHS` and `SEED` environment
overrides. A 100-epoch LaBraM experiment is substantially more work than ten-epoch fine-tuning;
the 16-hour request is not a runtime guarantee. If necessary, run separate `train --fold N` jobs
with outputs `OUTPUT/fold_N`, then use `summarize --output OUTPUT` once all ten finish.

`plot --output OUTPUT` generates the existing train/validation/test mean ± SD figures.
Run metadata and W&B record `initialization: random` and `checkpoint_sha256: null`. The comparison
signature prevents mixing random and pretrained folds. Complete older experiment folders remain
readable; do not combine their old configuration format with newly generated folds.

## Code and verification

`model.py:load_model` shares construction and head initialization between modes and conditionally
loads student weights. `training.py:resolve_model_options` resolves the starting profile while
honoring explicit overrides. `runner.py` skips checkpoint hashing for random runs and records the
initialization. Tests check seeded reproducibility, differing seeds, gradients, absence of checkpoint
reads, and actual CPU training/export for both pretrained and random LaBraM.
