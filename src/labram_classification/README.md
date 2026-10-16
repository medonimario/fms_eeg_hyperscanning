# Production LaBraM fine-tuning with Braindecode

This package implements **MirrorGame**, **ForceGame feedback and interaction**, and **Circling** binary classification with the downloaded `models/labram-base.pth`. It does not contact Hugging Face. Optional W&B logging supports online, offline, and disabled modes. Friendship classification is deferred until its original labeling script is provided.

A CPU-only band-power/logistic-regression comparator is available: see [BASELINES.md](BASELINES.md).

A native Braindecode **EEGNet-8,2** comparator trained from scratch is available via
`train --model eegnet` or `cv --model eegnet`: see [EEGNET.md](EEGNET.md) for its training
profile and HPC submission. It uses the same caches, folds, diagnostic curves and figure exports.
Commands without `--model` retain the LaBraM defaults documented below.

The introductory scripts are now in `src/labram_classification_lessons/`; production code does not import them. Run commands below from the repository root, with `PYTHONPATH=src`. The pipeline has four stages:

1. Read preprocessed EEG with MNE, preserve the dataset-specific operations below, and cache 4-second windows.
2. Assign whole dyads/triads to ten folds, keeping every participant, condition and overlapping window from a group together.
3. Fine-tune Braindecode's native `Labram` through `EEGClassifier`, using one explicit validation fold.
4. Restore the epoch with the best validation balanced accuracy, report final held-out test performance, and summarize all ten runs. Epoch-level test curves are diagnostic only.

## Install on the HPC

Use `src/labram_classification/requirements.txt` for a **new Linux x86_64, Python 3.11 environment with glibc 2.28 or newer**.
You create and install this environment on the cluster; no remote access by the coding agent is needed.

```bash
python3.11 -m venv /your/path/braindecode-env
source /your/path/braindecode-env/bin/activate
pip install -r src/labram_classification/requirements.txt
python -m pip check
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m labram_classification --help
```

The application pins are copied from the installed local environment, including **Braindecode 1.8.1**.
The GPU-specific exception is the matched `torch==2.6.0+cu118` / `torchaudio==2.6.0+cu118` pair from the
[official PyTorch CUDA 11.8 distribution](https://docs.pytorch.org/get-started/previous-versions/), selected
for the supplied V100/CUDA 11.8 job. Both packages are pinned together; the requirements file includes
their package index. This is an explicit platform adaptation, not a claimed freeze of an existing HPC
environment. The user's initial Braindecode 1.5 report differs from the local installed version; this
implementation has not been verified against 1.5.

The complete local `pip freeze` is preserved separately as `requirements-local-freeze.txt`.
`requirements-environment.json` records its interpreter/platform and successful local `pip check`.
That macOS snapshot is provenance, not the file to install on the cluster. It includes development
and unrelated installed packages; the HPC installation file lists the application's direct dependencies
and lets pip resolve their dependencies for Linux. Neither file replaces the root project's requirements.

Install where internet access is available. On an allocated GPU node, run the smoke command below
before full cross-validation. Package resolution cannot verify your cluster driver or GPU execution;
the CUDA environment has not been runtime-tested here. Loading `cuda/11.8` alone does not install PyTorch.
Once the installed HPC environment passes its checks, save its complete resolved snapshot:

```bash
python -m pip freeze > requirements-hpc-resolved.txt
```

Each training run also records the actual key library versions. Keep that resolved snapshot with your
experiment outputs to reproduce the transitive dependencies selected during installation.

## Prepare data once

Set `MG_DATASET_PATH`, `FG_DATASET_PATH`, and `CIRCLING_DATASET_PATH` in the repository `.env` or export them in the shell. Explicit `--data-dir /path` takes precedence. A path can point to the files themselves or their parent containing `preprocessed/`.

```bash
python -m labram_classification prepare --dataset mg \
  --contrast solo-coordination --cache data/processed/mg_solo_coordination

python -m labram_classification prepare --dataset fg \
  --cache data/processed/fg_feedback

python -m labram_classification prepare --dataset fg --fg-task interaction \
  --cache data/processed/fg_interaction

python -m labram_classification prepare --dataset circling \
  --cache data/processed/circling \
  --bad-segments /data/circling/dict_with_short_bad_epochs.pickle
```

Each cache must be a **new directory**. A failed preparation leaves an incomplete directory for diagnosis; choose a new destination when retrying. Training requires the completion marker `cache.json`. Preparation runs sequentially, holding one recording and its extracted windows in memory at a time. Do this on a suitable CPU compute node if the recordings are large.

MirrorGame alternatives are `solo-spontaneous` and `spontaneous-coordination`; create a separate cache/output directory for each contrast.

| Dataset/task | Class 0 | Class 1 | Group | Windowing |
| --- | --- | --- | --- | --- |
| MG solo-coordination (default) | Solo | Coordination | Dyad | Two 4 s windows per 6 s epoch, 2 s stride |
| MG solo-spontaneous | Solo | Spontaneous | Dyad | Same |
| MG spontaneous-coordination | Spontaneous | Coordination | Dyad | Same |
| FG feedback | Event name ending `n` | Other event names | Triad | Crop from −0.5 to 3.5 s, resample to 200 Hz |
| FG interaction | Solo: event name starts `T1P` | Interactive trio: event name starts `T3P` | Triad | Same FG crop/resampling; exclude other events |
| Circling | `closed1` | `open` | Dyad | Crop first 2 s, discard practice, slide 4 s windows with 2 s stride |

**MirrorGame:** filenames must match `s_101_Solo.set` (first digit is dyad member; last two are dyad). Require both members and both selected conditions per dyad. Add the average-reference projector without applying it, matching the source script; resample the full epoch, reorder the 64 channels, then slice `[0:800]` and `[400:1200]`. Already-active references remain active. The pending projector is intentionally not an additional rereference.

**FG:** leading digits identify the triad and the following letter identifies the participant, e.g. `301A...-epo.fif`. Event labels reproduce the supplied script's suffix rule. No friendship labels are inferred. MNE's inclusive crop can produce **801** samples at 200 Hz. Default `--fg-endpoint strict` stops if this occurs. After inspecting the source, rerun preparation with `--fg-endpoint trim` to explicitly retain samples 0–799 (−0.5 through 3.495 s). This is recorded in cache metadata; it is an explicit correction for the inclusive endpoint, not an unreported reproduction change.

**FG interaction:** select `--fg-task interaction` to reproduce `make_FG_group_folds.py`:
retain only event names beginning `T1P` (solo, class 0) or `T3P` (interactive trio, class 1).
Solo feedback depends on one person's force; trio feedback depends on the three-person mean.
The prefix determines the target regardless of the feedback suffix: both feedback types are retained.
Filtering precedes cropping/resampling, and `source_epoch` retains the original epoch index.
Files without eligible events are reported and skipped. `files.csv` still lists all discovered files,
and fold assignment uses all discovered triads before filtering, matching the original script.
An entirely empty selection fails; training also rejects any partition lacking either class.
`--fg-task feedback` remains the default and keeps its existing suffix labels and event selection.
Use a separate cache for each task: `cache.json` records `dataset: fg` and `contrast: interaction`
or `feedback`, and this metadata is carried into both classifiers' run configuration and W&B.

To run both models on the interaction task (after preparing its cache):

```bash
python -m labram_classification cv \
  --cache data/processed/fg_interaction --output runs/fg_interaction_labram --device cuda
python -m labram_classification baseline \
  --cache data/processed/fg_interaction --output runs/fg_interaction_bandpower
python -m labram_classification baseline --feature-mode time-bandpower \
  --cache data/processed/fg_interaction --output runs/fg_interaction_time_bandpower
```

Each command runs all ten folds; choose a new output directory for every experiment. For HPC jobs,
point `CACHE_DIR` at the new interaction cache and give `OUTPUT_DIR` and `WANDB_RUN_GROUP` distinct
interaction names. The existing GPU fine-tuning and CPU baseline job scripts work unchanged.
The task is selected during preparation; no task flag or label changes are needed during training.

**Circling:** `pair<digits>` identifies the group. After cropping at 2 s, discard the first 6 practice epochs (12 for pair2030). A bad dictionary index identifies a 6 s section: `trial=index//3`, section `index%3`. Exclude every candidate window with any overlap, then resample each surviving window to 200 Hz. This order matches the original script. Missing keys mean no listed exclusions, as before; a missing dictionary file is an error. Only load a trusted local pickle. The participant column retains the source recording identifier; dyad grouping is the authoritative leakage barrier.

All outputs have 64 ordered EEG channels and 800 samples. MNE positions/montages are not supplied as model features: LaBraM uses the explicit channel names to select its learned positional embeddings. The unrelated EEGPT montage deprecation warnings during Braindecode import do not change these inputs.

Caches store **float32 microvolts** in one `.npy` array per source recording, plus `files.csv`, `windows.csv`, and `cache.json`. The loader divides by **100** immediately before model input, matching the original training engine (its pickle maker only performed volts → microvolts). It never divides twice. Memory mapping avoids loading the entire dataset into RAM and avoids millions of per-window pickles. This thin PyTorch dataset adapter is accepted directly by Braindecode's classifier; MNE supplies preprocessing and Braindecode supplies the model/training interface.

## Splits and fine-tuning

Sorted group IDs are shuffled once with NumPy RandomState seed42, then divided with `KFold(10, shuffle=False)`, as in the original makers. For run `k`, test is fold `k`, validation is `(k+1)%10`, and training is the other eight. The training seed is separately **0**. Assertions check disjoint groups and participants and require both classes in every partition. Changing available groups changes the folds; compare `files.csv` against your previous manifests when reproducing historical experiments.

```bash
python -m labram_classification train \
  --cache data/processed/mg_solo_coordination --fold 0 \
  --output runs/mg_solo_coordination/fold_0 --device cuda

# Or all ten sequentially, in a NEW output directory:
python -m labram_classification cv \
  --cache data/processed/fg_feedback --output runs/fg_feedback --device cuda

python -m labram_classification summarize --output runs/fg_feedback
```

| Setting | Default |
| --- | --- |
| Model | Base LaBraM, 12 blocks, 200-dimensional tokens, mean pooling |
| Initialization | Local original student encoder; fresh binary head and mean-pooling normalization |
| Input | `(batch, 64, 800)`, microvolts / 100 |
| Loss | One-logit BCEWithLogitsLoss, no class weighting |
| Epochs / warmup | 10 / 5 |
| Batch / accumulation | 128 / 2; effective batch 256 |
| LR | MG 1e−4; FG and Circling 5e−4 |
| Schedule | Inclusive linear warmup from 0, then per-update cosine toward 1e−6 |
| Optimizer | AdamW, betas (0.9, 0.999), epsilon 1e−8 |
| Layer LR decay / weight decay | 0.65 / 0.05 |
| DropPath | 0.2 |
| Position/QKV | Absolute positions enabled, no relative-position bias, no QKV bias |
| Precision | CUDA autocast + GradScaler; CPU float32 |
| Selection | Highest validation balanced accuracy, first epoch on ties |

All encoder weights are fine-tuned. No mixup, augmentation, normalization fitted across recordings, early stopping, gradient clipping, or EMA is added. Incomplete batches and incomplete accumulation groups are dropped during training, matching the original loop; validation/test retain every window. There must be at least one complete optimizer update per epoch. Layer IDs match the original optimizer, including its treatment of the temporal embedding as the final layer. Biases, vectors, and positional/class tokens receive no weight decay.

The checkpoint adapter renames original student parameters, trims the temporal table for four patches, drops pretraining-only heads, and strictly checks all remaining tensor shapes. The original fine-tuning loader did **not** map the student's `norm` into `fc_norm`; production likewise initializes `fc_norm` to ones/zeros. This differs intentionally from the introductory lesson's conversion. The classification head uses truncated-normal initialization scaled by 0.001. Braindecode stores the full 128-channel positional table, while forward calls select the explicit 64 recorded channel names. Local checkpoint loading does not eliminate the need for parameter-name and temporal-table conversion.

This preserves the intended experimental protocol, not guaranteed bitwise identity across frameworks/devices. Train and test partitions are also evaluated after every epoch for diagnostic curves; only validation balanced accuracy selects checkpoints. Outputs are window-level; overlapping windows are correlated and are not independent subjects. Fold mean and sample SD are descriptive, not an independence-based significance test.

## Outputs and recovery

Each `fold_k/` contains:

- `config.json`: options, cache metadata/hash, checkpoint SHA256, library versions and update count.
- `train_windows.csv`, `valid_windows.csv`, `test_windows.csv`: exact partition provenance.
- `best_model.pt`: validation-selected model **state dictionary**, including wrapper `backbone.*` names.
- `history.json`: per-epoch scores/losses and per-batch losses, saved atomically after every completed epoch.
- `epoch_metrics.csv`: flat per-epoch optimization, evaluation, and timing measurements, also saved every epoch.
- `valid_predictions.csv`, `test_predictions.csv`: probabilities, predictions, labels and provenance.
- `result.json`: completion marker, selected epoch and metrics.

Final metrics include binary cross-entropy loss, accuracy, balanced accuracy, ROC AUC, trapezoidal PR AUC, average precision, F1 and a `[0,1]` confusion matrix. PR AUC and average precision are named separately because they are different summaries. Threshold is strictly greater than 0.5, matching two-column argmax's class-0 tie. No threshold is tuned on test data.

Summarization requires all ten completed, non-smoke folds with matching configurations; it writes `fold_metrics.csv` and `summary.json`. Caches are portable because array paths are relative; source paths remain provenance and are not opened during training. Metadata has a SHA256 integrity check; signal arrays are not individually hashed. Preserve the cache immutably throughout an experiment.

Existing output directories are never overwritten. There is **no mid-epoch optimizer resume**. Following an interrupted CV job, retain completed folds, move the incomplete fold directory aside, run the remaining folds individually with exactly the same options, then summarize. `best_model.pt` alone is not a resumable optimizer checkpoint.

To restore a trained model for inference:

```python
import torch
from labram_classification.model import load_model
model = load_model()  # local pretrained file supplies the architecture initialization
model.load_state_dict(torch.load("runs/task/fold_0/best_model.pt", map_location="cpu", weights_only=True))
model.eval()
# x must already have the documented channel order, shape, and microvolts/100 scaling.
with torch.no_grad():
    probabilities = model(x).sigmoid()
```

## HPC execution

Prepare/cache the data first. Copy the cache and local checkpoint to cluster storage; the raw recordings are unnecessary for training. Create log directories **before submission**, because LSF opens its output files before the script runs.

```bash
mkdir -p outputs errors
export PROJECT_DIR=/absolute/path/fms_eeg_hyperscanning
export VENV=/absolute/path/braindecode-env
export CACHE_DIR=/absolute/path/cache/fg_feedback
export OUTPUT_DIR=/absolute/path/results/fg_feedback_run1
bsub < src/labram_classification/hpc/train_cv.lsf
```

The supplied Bash LSF script retains your V100/CPU/memory/wall-time requests and runs all ten folds sequentially. Adjust the resource request and time limit after measuring a fold; 16 hours and memory sufficiency have not been established on your data. Batch128 accumulation2 preserves the requested effective batch; if GPU memory is insufficient, explicitly change the settings and record the changed protocol. There is no embedded API key, username or email. Local result files are always saved. The job defaults to offline W&B logging; export `WANDB_MODE=online` for a live dashboard or `WANDB_MODE=disabled` to turn tracking off.

First test a small run on the allocated GPU:

```bash
python -m labram_classification train --cache "$CACHE_DIR" \
  --output /absolute/path/results/smoke --fold 0 --device cuda \
  --batch-size 2 --update-freq 2 --epochs 1 --warmup-epochs 0 \
  --num-workers 0 --smoke-windows-per-class 2
```

Smoke subsets retain the existing group partitions but are not performance estimates. They are marked in results and refused by CV summarization. Use `--device cpu` for local diagnostics.

## W&B progress tracking

The captured local environment already includes W&B. In a separate environment, install it with `pip install wandb`, validate the environment, and capture its own freeze. For live progress, authenticate once using `wandb login` on the cluster login node, or supply `WANDB_API_KEY` through the job environment. Do not put the key in code or the submission script.

```bash
export WANDB_MODE=online
export WANDB_PROJECT=labram-finetuning
export WANDB_RUN_GROUP=mg_solo_coordination_run1
# Optional when using a team workspace:
# export WANDB_ENTITY=your-team
python -m labram_classification cv \
  --cache data/processed/mg_solo_coordination \
  --output runs/mg_solo_coordination_run1 --device cuda
```

The same exports work before `bsub < src/labram_classification/hpc/train_cv.lsf`. CLI flags `--wandb-mode`, `--wandb-project`, `--wandb-entity`, and `--wandb-group` override environment values. Direct Python commands default to disabled if `WANDB_MODE` is unset; the HPC script defaults it to offline. Each fold creates a separate run named `<group>-fold-<k>`. Set a unique group for each experiment; otherwise the output directory's parent name is used. Smoke runs are identified by job type.

Logged data:

- Every 20 optimizer updates (configurable with `--wandb-log-every`) and the final update of each epoch: accumulated-batch training loss and minimum/maximum layer learning rate.
- Each epoch: optimization training loss; evaluation-mode train/validation/test loss, accuracy and balanced accuracy; training/validation duration and additional diagnostic evaluation duration.
- After restoring the best checkpoint: selected epoch, final validation/test metrics, and confusion matrices in the run summary.
- Run configuration: training options, fold, cache configuration, checkpoint hash and library versions.

Epoch test metrics are display-only; checkpoint selection and learning-rate scheduling never consume them. Final test metrics are recomputed with the validation-selected checkpoint. Raw EEG, per-window prediction files and checkpoint files are not explicitly uploaded by this integration. Local files remain available as before. Each run is finished before starting the next fold; training errors or keyboard interruptions mark its W&B run as failed. Tracking errors propagate rather than silently disabling your requested logging.

For a cluster without outbound connectivity, use `WANDB_MODE=offline`. W&B stores its run under each fold's `wandb/` directory. Later, from a machine with connectivity and W&B authentication, synchronize each recorded offline run:

```bash
wandb sync /absolute/path/results/experiment/fold_0/wandb/offline-run-*
```

Offline mode has no live remote dashboard until synchronization. See the [W&B network/offline guidance](https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues). Neither `.env` contents nor API keys are added to the run configuration; authentication uses W&B login or exported environment variables.

## Live curves and final figures

New training runs automatically log these additional W&B keys:

- `epoch/train_eval_loss`, `epoch/train_eval_acc`, `epoch/train_eval_balanced_accuracy`
- `epoch/test_loss`, `epoch/test_acc`, `epoch/test_balanced_accuracy`
- `epoch/diagnostic_duration`

The existing `epoch/train_loss` is the loss observed while optimizing the model with DropPath active and changing weights. The `train_eval_*` metrics instead use a separate epoch-end pass with fixed weights, DropPath disabled, no gradients, and **all training windows**, including the incomplete tail batch. These measurements are comparable to validation and test metrics. Loss is weighted by the number of windows, not by the number of batches. The extra train/test passes add runtime but no backward passes. They restore model modes and PyTorch RNG states to avoid changing subsequent training randomness. `epoch/dur` retains the original training/validation timing; `diagnostic_duration` measures the extra evaluation and does not include W&B upload time.

Test curves never select a checkpoint or affect the optimizer/schedule. Continue using validation for experimental decisions; the final test result uses each fold's validation-selected checkpoint, not whichever epoch had the highest test score. Training and test diagnostics are saved even when W&B is disabled.

After all ten folds finish, generate publication figures from local files:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m labram_classification plot \
  --output models/results/mg_solo_coordination_run2
```

`--output` identifies the experiment containing `fold_0` through `fold_9`. The default destination is its `figures/` directory; override with `--figures-dir /path/to/figures`. It contains:

- `learning_curves.pdf`, `learning_curves.svg`, `learning_curves.png`: three panels (BCE loss, accuracy, balanced accuracy), each showing train, validation and test.
- `learning_curves_folds.csv`: individual fold/epoch/split measurements.
- `learning_curves_summary.csv`: equal-weight fold means, **sample SD (ddof=1)**, and fold counts.

The shaded regions are mean ± SD across ten folds, **not confidence intervals**. Train curves use `train_eval_*`, not optimization-mode loss. Bands are not clipped at metric boundaries. The command also refreshes the standard final-score summary and rejects smoke runs, incomplete runs, incompatible configurations, missing metrics, and mismatched epoch counts. It does not contact W&B or require the EEG cache/GPU. Figures can be regenerated without retraining when all necessary histories exist.

Older runs lack per-epoch train accuracy and test metrics; these cannot be recovered from a single best checkpoint. The plotting command reports missing metrics rather than inventing values. Already-running jobs will not adopt this change: use a fresh output directory for a new run after updating the cluster checkout. Existing epoch histories remain untouched by plotting.

## File guide and checks

| File | Responsibility |
| --- | --- |
| `__main__.py` | CLI arguments and command dispatch |
| `baseline.py` | Whole-window/time-resolved band-power extraction and grouped logistic-regression baseline |
| `data.py` | Discovery, MNE preprocessing, disk cache, group assignments and dataset adapter |
| `model.py` | Offline checkpoint conversion and channel-aware binary LaBraM wrapper |
| `training.py` | EEGClassifier subclass, layer-specific AdamW groups, accumulation, AMP and schedule |
| `monitoring.py` | Read-only train/test epoch evaluation and durable curve logging |
| `plotting.py` | Ten-fold mean/SD curve aggregation and PDF/SVG/PNG/CSV exports |
| `tracking.py` | Optional W&B lifecycle, progress callbacks and final metric logging |
| `runner.py` | Fold orchestration, manifests, selection, predictions, metrics and summary |
| `hpc/train_cv.lsf` | Offline sequential CV job template |
| `requirements.txt` | Linux/Python 3.11 installation requirements with matched CUDA 11.8 Torch/TorchAudio |
| `requirements-local-freeze.txt` | Unmodified full local pip freeze, retained as provenance |
| `requirements-environment.json` | Snapshot interpreter/platform provenance and dependency-check result |

```bash
python -m pytest tests/labram_classification tests/labram_classification_lessons
python -m ruff check src/labram_classification tests/labram_classification
```

Regression tests exercise synthetic MNE data, overlap, endpoint handling, bad segments, group isolation, unit scaling, accumulated optimizer equivalence, and the actual local checkpoint's forward/backward path. Synthetic tests do not establish correctness of unknown real filenames/event mappings or GPU performance. Review the first real prepared manifests and run the smoke command before full HPC CV.

API references: [Braindecode Labram](https://braindecode.org/stable/generated/braindecode.models.Labram.html), [EEGClassifier](https://braindecode.org/stable/generated/braindecode.EEGClassifier.html), [skorch predefined_split](https://skorch.readthedocs.io/en/stable/helper.html#skorch.helper.predefined_split), [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

### Verification performed during implementation

The 12 production and lesson tests passed locally, including an end-to-end CPU fit using the downloaded checkpoint on synthetic windows. Ruff and the LSF script's Bash syntax check passed. The full repository test collection is currently blocked by the unrelated template module's missing `typer` dependency in the Braindecode environment. A direct numerical comparison with the old model could not run because its `timm` dependency is absent; bitwise parity is not claimed. The real dataset drive was unavailable, so full-file preparation, real event/filename coverage, V100 AMP, and complete ten-fold training remain to be verified on the user's data and cluster.

W&B addition: all five tracking tests passed, including a real offline SDK run with W&B 0.30.0, alongside the 12 existing targeted tests. The real SDK test needs permission to start its local background service/socket. Online authentication and upload were not exercised during implementation.

HPC requirements validation: pip successfully resolved the Linux target wheels, including the Linux-only NVIDIA CUDA and Triton dependencies. All active Linux dependency specifiers were checked against the resolved versions; see `requirements-linux-check.json`. This is a dependency-metadata check, not a GPU runtime test.
