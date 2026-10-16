# EEGNet comparator

Use `--model eegnet` with the existing `train` or `cv` commands to train Braindecode's
native EEGNet from scratch. This works with every prepared cache: all MirrorGame contrasts,
FG feedback, FG interaction, and Circling. No new preparation, checkpoint, download, or package
installation is required in the documented Braindecode 1.8.1 environment.

## Model and inputs

`model.py:BinaryEEGNet` wraps `braindecode.models.EEGNet` (the 2018 architecture, formerly named
EEGNetv4) and returns one raw logit per window for the shared binary cross-entropy loss.
The configuration is EEGNet-8,2: F1=8 temporal filters, depth multiplier D=2, F2=16 pointwise
filters, temporal kernel length 64 samples (320 ms at 200 Hz), separable temporal kernel 16,
and mean-pooling factors 4 and 8. Spatial max norm is 1; the final linear head has max norm 0.25.
Batch normalization, ELU and dropout use the native model implementation.

Inputs retain the same 64-channel ordering and 800 samples as LaBraM. The shared loader supplies
microvolts divided by 100. This fixed scale preserves amplitude differences and requires no fitted
normalization statistics. It is an explicit comparison choice, not a claim that this scale is optimal
for EEGNet. There is no per-window z-scoring, added filtering, or montage conversion. Batch-normalization
running statistics update only during training; validation, test and diagnostic passes use evaluation mode.

References: [Braindecode EEGNet API](https://braindecode.org/stable/generated/braindecode.models.EEGNet.html)
and [Lawhern et al., EEGNet](https://arxiv.org/abs/1611.08024).

## Starting training profile

| Setting | EEGNet default | Existing LaBraM default |
| --- | --- | --- |
| Initialization | Random each fold, seeded | Local pretrained backbone, fresh head |
| Epochs | 100 | 10 |
| Batch size | 64 | 128 |
| Gradient accumulation | 1 | 2 |
| Initial learning rate | 1e-3, all datasets | MG 1e-4; FG/Circling 5e-4 |
| Warmup epochs | 0 | 5 |
| Layer decay | 1, uniform learning rate | 0.65 |
| Weight decay | 0 | 0.05 |
| Dropout / stochastic depth | Dropout 0.5 | Stochastic depth 0.2 |

These are starting settings for this benchmark, not tuned results or an exact reproduction of the
EEGNet paper's training protocol. EEGNet uses the shared AdamW implementation (with zero weight decay
by default), cosine learning-rate schedule toward 1e-6, and optional CUDA mixed precision. The shared
trainer uses complete batches and accumulation groups; training tails are omitted from optimizer
updates, while epoch-end diagnostics and final predictions include every window.

The run completes the requested number of epochs, without early stopping, then restores the checkpoint
with highest **validation balanced accuracy** (first epoch wins ties). Test curves never select the
checkpoint. Use validation performance and convergence to assess whether 100 epochs is sufficient;
do not tune against the displayed test curves.

Explicit CLI arguments override defaults. EEGNet rejects LaBraM-specific nonzero `--drop-path` or
`--layer-decay` other than 1, so copying the old fine-tuning command cannot silently apply those
settings. Use `--eegnet-dropout` for EEGNet dropout. `--checkpoint` is unused for EEGNet.

## Run locally or on an allocated compute node

Run from the repository root with the environment activated:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Short connectivity/training check; intentionally excluded from CV summaries:
python -m labram_classification train --model eegnet \
  --cache /path/to/prepared/cache --output runs/eegnet_smoke --fold 0 \
  --epochs 2 --batch-size 4 --smoke-windows-per-class 8 \
  --num-workers 0 --device cpu --wandb-mode disabled

# Ten folds, each initialized independently:
python -m labram_classification cv --model eegnet \
  --cache /path/to/prepared/cache --output runs/eegnet_cv \
  --device cuda --wandb-mode online --wandb-group eegnet_cv

python -m labram_classification plot --output runs/eegnet_cv
```

Use a new output directory per experiment. `train --fold N` also permits submitting one fold at a time;
use `OUTPUT/fold_N` as its output and `summarize --output OUTPUT` once all ten complete.
Keep the options and W&B group consistent across folds.

## HPC submission

`hpc/eegnet_cv.lsf` uses the same V100/CUDA 11.8 environment setup as the existing fine-tuning job.
It runs all ten folds sequentially; its wall-time request is a starting allocation, not a runtime guarantee.
Create LSF log directories in the submission directory before submission:

```bash
mkdir -p outputs errors
export PROJECT_DIR="$HOME/FMs_EEG_hyperscanning/fms_eeg_hyperscanning"
export VENV=/work3/s204684/Thesis/braindecode-env
export CACHE_DIR=/work3/s204684/Thesis/data/processed/mg_solo_coordination
export OUTPUT_DIR="$PROJECT_DIR/models/results/mg_eegnet_run1"
export WANDB_MODE=online
export WANDB_PROJECT=labram-finetuning
export WANDB_RUN_GROUP=mg_eegnet_run1
bsub < src/labram_classification/hpc/eegnet_cv.lsf
```

Change the cache and output/group names for FG or Circling. `EPOCHS` (default 100) and `SEED`
(default 0) can be exported before submission. Existing W&B authentication is reused.

## Implementation and outputs

- `model.py`: native EEGNet construction and binary-output adapter.
- `training.py`: resolves model-specific defaults; shared optimizer, selection and diagnostics.
- `__main__.py`: exposes the model choice and validates CLI options.
- `runner.py`: initializes EEGNet without reading or hashing LaBraM weights; records model identity,
  initialization, parameter count, fixed input scale, resolved options and library versions.
- `tracking.py`, `monitoring.py`, `plotting.py`: shared W&B metrics, durable epoch logs and ten-fold plots.

The existing best-model weights, validation/test predictions, per-fold metrics, CV summaries,
and train/validation/test mean ± sample SD plots all apply. No EEGNet-specific plot command is needed.
Model identity is included in the comparison signature, preventing mixed-model fold summaries.
The new run metadata also means older and newer LaBraM fold configurations should not be mixed
within one CV experiment; complete existing experiments remain readable.
