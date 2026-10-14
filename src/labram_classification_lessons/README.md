> These are the archived learning scripts. Use [the production pipeline](../labram_classification/README.md) for full experiments.

# Learning Braindecode with LaBraM

This directory will contain a step-by-step migration of the existing binary
classification experiments. Implementation will proceed one lesson at a time.
Lesson 1 implements inspection and the original MirrorGame window extraction.
Lesson 2 adds Braindecode datasets and dyad-level folds. Training follows later.

## Experiment contract recovered from the reference scripts

| Dataset | Class 0 | Class 1 | Split unit | Examples |
| --- | --- | --- | --- | --- |
| Circling | closed1 | open | Dyad | 4-second windows, 2-second stride |
| FG | Condition name ends in `n` | Other condition names | Triad | 4-second interval, -0.5 to 3.5 seconds |
| MirrorGame (default contrast) | Solo | Coordination | Dyad | Two 4-second windows per 6-second epoch |

The existing scripts target 200 Hz and a common ordered 64-channel list.
Validate actual channel names and sample counts rather than assuming compliance.
FG starts at -0.5 seconds and is cropped through 3.5 seconds, as confirmed by
the user. Verify the sample count when loading: inclusive endpoints can add a
sample, even though the intended interval is four seconds.

Circling excludes practice trials (12 for pair 2030, otherwise 6), crops from
2 seconds, and rejects windows overlapping externally specified bad intervals.
Preserve trial identities when reproducing this rejection logic.
MirrorGame adds an average-reference projector; inspect whether it is applied
before deciding how to reproduce referencing.

The old pipeline exports microvolt-valued arrays and divides them by 100 in
the training/evaluation loop. Keep MNE objects in volts; explicitly convert at
the model boundary. Matching the old numerical scaling means volts times 1e4.

Ten folds are assigned by group with seed 42. For run k, the test fold is k,
validation is (k + 1) modulo 10, and the other eight folds are training.
For exact comparisons, reuse or verify existing group-to-fold assignments.
All participants and overlapping windows from a dyad/triad stay together.
Check whether any participant appears in multiple groups before trusting this
as a participant-independent evaluation.

## Lessons and completion checks

1. Inspect one original MNE epoch file. Report event names, channel names,
   sample rate, start/end times, shape, and group identity. Choose and record
   the Python environment and Braindecode version before implementation.
2. Implement dataset-specific loading and explicit binary labels. Retain
   source file, group, participant where available, trial, and window metadata.
3. Reproduce preprocessing and windows. Use Braindecode's
   `create_from_mne_epochs` where appropriate; Circling also needs its custom
   bad-interval rejection. Verify one window numerically against the old pipeline.
4. Implement group-disjoint train/validation/test splits. Assert disjointness
   and report class counts before training. Supply the validation dataset
   explicitly instead of using the classifier's default split.
5. Load pretrained `braindecode.models.Labram`. Verify checkpoint provenance,
   parameter loading, channel-position mapping, pooling, input scaling, and
   patch dimensions. A plain constructor alone is not pretrained fine-tuning.
6. Run one batch, then a tiny training smoke test with `EEGClassifier`.
   Choose the loss/head convention explicitly: the old implementation uses
   one logit with BCEWithLogitsLoss; a two-logit CrossEntropyLoss alternative
   remains binary classification but changes the parameterization.
7. Train one fold, select the checkpoint by validation balanced accuracy,
   restore that checkpoint, and evaluate the held-out test fold.
8. Run all folds and save predictions, group identities, configuration,
   checkpoint information, and metrics. Match PR-AUC definitions explicitly;
   average precision and trapezoidal PR area are not interchangeable.
9. Add other models behind the same experiment interface. Preserve the data
   and evaluation protocol while making model-specific requirements explicit.

Overlapping input windows do not require Braindecode's `cropped=True` mode.
Our initial design predicts one label per window, using `cropped=False`.
Window-level metrics and trial/group-level metrics answer different questions;
define any aggregation before evaluating results.

## API references

The initial reference documentation identifies itself as Braindecode 1.5.1.
The stable URLs can change; verify APIs against the installed version.

- [MNE epoch dataset builder](https://braindecode.org/stable/generated/braindecode.datasets.create_from_mne_epochs.html)
- [LaBraM](https://braindecode.org/stable/generated/braindecode.models.Labram.html)
- [EEGClassifier](https://braindecode.org/stable/generated/braindecode.classifier.EEGClassifier.html)

## Lesson 1: inspect MirrorGame

From the repository root, run:

```bash
python src/labram_classification_lessons/inspect_mirrorgame.py
python src/labram_classification_lessons/inspect_mirrorgame.py --file s_101_Spontaneous.set
```

This uses MNE, NumPy, and python-dotenv. The repository `.env` supplies
`MG_DATASET_PATH`; an existing environment variable takes precedence. The path
may point directly to preprocessed files or to a parent with `preprocessed/`.
Source files are read only; no windows are written to disk.

The current inventory contains 80 files per condition (Solo, Spontaneous,
Coordination). The default contrast now retains Solo=0 and Coordination=1. All three pairwise
contrasts are selectable through `--contrast` in both lesson scripts. Filename `s_103_Solo.set` identifies participant 103
in dyad 3; participant 203 belongs to the same dyad. Actual fold assignment is
the next lesson, not part of this inspection script.

The first inspected Solo file has 88 epochs, 64 channels, and 1536 samples at
256 Hz. Its first sample is at -1 second. Resampling the whole six-second epoch
to 200 Hz precedes slicing, exactly as in the original code:

```text
MNE epoch:                  (64, 1200), volts
get_data(units="uV"):       (64, 1200), microvolts
first slice [0:800]:         (64, 800)
second slice [400:1200]:     (64, 800)
```

The two windows start 0 and 2 seconds from the beginning of each stored epoch,
not necessarily at event time 0. For this file they start at -1 and +1 seconds
on the MNE time axis. Their shared 400 samples must be exactly equal. Keeping
the epoch axis gives `(n_epochs, 2, 64, 800)` for inspection; the eventual
training dataset will expose one `(64, 800)` window per item and retain its
participant, dyad, source epoch, start sample, and label as metadata.

No additional filtering or ICA is performed on already preprocessed files.
The original `projection=True` call adds an inactive average-reference projector;
it does not apply it. We preserve that behavior. The source FDT name suggests
prior average referencing, but naming alone does not establish preprocessing
history. When constructing Braindecode windows, verify that this inactive
projector does not get applied implicitly and change the numerical inputs.

Unlike the original script's warning-and-continue behavior, missing required
channels raise an error here instead of silently yielding a different ordering.

The inspection environment currently has MNE 1.10.2. The separate
`braindecode-env` has Braindecode 1.8.1 and MNE 1.13.2; python-dotenv is now available.
Reconcile the environment and verify numerical parity before training; do not
assume the supplied 1.5.1 API PDF describes the installed release.


## Lesson 2: Braindecode datasets and dyad splits

Use the Python environment containing Braindecode (verified with 1.8.1 and MNE
1.13.2). The default contrast is now Solo vs Coordination in both scripts.

| CLI contrast | Class 0 | Class 1 |
| --- | --- | --- |
| `solo-coordination` | Solo | Coordination |
| `solo-spontaneous` | Solo | Spontaneous |
| `spontaneous-coordination` | Spontaneous | Coordination |

```bash
python src/labram_classification_lessons/mirrorgame_dataset.py --manifest-only
python src/labram_classification_lessons/mirrorgame_dataset.py --contrast solo-coordination --fold 0
python src/labram_classification_lessons/mirrorgame_dataset.py --contrast solo-spontaneous --fold 3
```

The default run loads only one complete dyad from each partition as a learning
preview. Fold assignments are computed from all 40 dyads before subsetting.
Use `--all` to load all 160 selected recordings; this keeps windows in memory
and requires several GB of RAM. Nothing is trained or written to disk.

### What replaces the pickle files?

```text
SET/FDT files -> MNE Epochs -> create_from_mne_epochs -> BaseConcatDataset
                                                     | train
                                                     | valid
                                                     | test
```

`WindowsDataset` provides indexed access to windows. `BaseConcatDataset`
combines these child datasets. The installed epoch builder creates one child
per source epoch, containing its two windows. `dataset[i]` returns:

- `X`: float32 array, shape `(64, 800)`, already scaled to microvolts/100.
- `y`: integer binary label determined by the filename's condition.
- Crop indices: Braindecode's window-index/start/stop bookkeeping. Its built-in
  start/stop include the source event sample offset; use our explicit
  `start_sample` and `stop_sample` columns for offsets within the stored epoch.

`dataset.get_metadata()` joins window metadata with descriptions, exposing
participant, dyad, condition, path, source epoch, fold, partition, and target.
Later, EEGClassifier will manage a PyTorch DataLoader to batch these items.
We do not need one pickle per window or our own Dataset subclass.
This initial implementation uses RAM; persistent caching can be added later.

### Who decides the dyad split?

Our `make_manifest` function assigns sorted, seed-42-shuffled dyads using the
same KFold procedure as the original script. Run k uses fold k for testing,
(k+1)%10 for validation, and the other eight folds for training. With 40 dyads:

| Partition | Dyads | Participants | Files |
| --- | --- | --- | --- |
| Train | 32 | 64 | 128 |
| Validation | 4 | 8 | 16 |
| Test | 4 | 8 | 16 |

The resulting partition name is attached to every child dataset. Braindecode's
`combined.split("split")` groups children by that description field. Thus we
choose the scientific grouping rule and Braindecode applies it. Assertions
check that neither dyads nor participants occur in multiple partitions.
Incomplete dyad/condition coverage raises an error instead of silently changing
the experiment. Window counts can vary across partitions due to cleaning.

### Numerical compatibility

The epoch builder creates new MNE epochs, which could apply pending projectors.
`windows_from_epochs` removes inactive projectors from a copy first, preserving
the supplied signal and the original MNE object. The builder performs only
window extraction, with 800 samples and a stride of 400. Every event code in a
recording maps explicitly to its filename-derived binary target.

MNE data stay in volts. A dataset transform converts each retrieved float32
window to microvolts/100. Do not divide by 100 again during training. Because
Braindecode casts volts to float32 before scaling, tiny rounding differences
from the old microvolt-first cast are expected and tested with tolerance.

The tests in `tests/labram_classification_lessons/test_mirrorgame_dataset.py` verify window signal/provenance,
projector preservation, all three contrasts across all ten folds, and rejection
of identity leakage and missing files. They use synthetic data, not the drive.

Read `make_manifest`, `windows_from_epochs`, and `build_datasets` in that order.
The next lesson will instantiate pretrained LaBraM and pass one batch through
it before introducing the training loop.


## Scaling and montage clarification

The original dataset maker exported microvolts, but the original
`engine_for_finetuning.py` divided inputs by 100 in both training (line 64)
and evaluation (line 173). Thus a 50-microvolt signal reaches the model as 0.5.
Our dataset transform reproduces both steps, not only the unit conversion.
This is an experiment-specific amplitude scaling, not a universal Braindecode
requirement or per-window standardization.

The `standard_1020` deprecation warnings originate from Braindecode's EEGPT
module during import. It creates its own channel lists and interpolation target
metadata; these calls do not set the montage on our MirrorGame epochs.
The current loading/windowing code preserves the channel locations read from
EEGLAB and does not interpolate to standard_1020. BioSemi64 remains relevant
for spatial analysis and plotting; do not replace existing coordinates merely
to silence an unrelated model's import warning. The native LaBraM adapter will
need explicit channel-name mapping to its learned positional embeddings.

Unit tests now live under `tests/labram_classification_lessons/`. Pytest is configured
to import packages from `src`, and the dataset module supports both package
imports and the existing direct script command. With pytest installed in the
Braindecode environment, run `pytest tests/labram_classification_lessons/`.

## Lesson 3: pretrained model and one batch

`mirrorgame_dataset.py` normally creates a manifest AND loads preview datasets.
Only `--manifest-only` stops after the manifest. Python functions are reusable:
`check_labram.py` calls them directly, so you do not need to run lesson 2 first.
The previous process's in-memory datasets do not persist after it exits.

```bash
python src/labram_classification_lessons/check_labram.py --contrast solo-coordination --fold 0
```

The script loads one complete TRAINING dyad and selects one window per class.
Validation and test signals are not loaded for this model check. A standard
PyTorch DataLoader collates the two dataset items into a batch. Later,
EEGClassifier will create these loaders for training and validation.

```text
Dataset item:       X=(64, 800), y=0 or 1
Batch:              X=(2, 64, 800), y=(2,)
LaBraM patches:     4 one-second patches per channel, created internally
Binary head:        1 raw logit per window
Wrapper output:     logits=(2,)
BCEWithLogitsLoss:   logits and float32 targets -> scalar loss
sigmoid(logits):    probability of class 1, for interpretation only
```

Do not apply sigmoid before BCEWithLogitsLoss: that loss performs the required
operation internally. Do not scale the input again; the dataset already scales
volts to microvolts/100. A new linear head must learn the selected contrast.
Its initial probabilities near 0.5 are not validation results.

### Local checkpoint and model adaptations

The only checkpoint source is `models/labram-base.pth`, resolved relative to
this repository, not your shell's current directory. There is no Hugging Face
request, download cache, or `--offline` flag in lesson 3 anymore. On the HPC,
copy this file along with the project, or specify its location:

```bash
python src/labram_classification_lessons/check_labram.py --checkpoint /path/to/labram-base.pth
```

`load_model` follows a simple sequence: read student weights, translate their
names, load the backbone strictly, then create the binary head. The two small
helpers isolate checkpoint-format details from the forward-pass lesson.

The downloaded file contains `model`, optimizer/scaler state, and other training
metadata. Within `model`, `student.*` contains the encoder we need. The loader
uses `weights_only=True` with a small allowlist for the saved NumPy scalar and
argument metadata; it does not use unrestricted pickle loading.

| Original checkpoint | Braindecode |
| --- | --- |
| `student.pos_embed` | `position_embedding` |
| `student.time_embed` | `temporal_embedding` |
| `student.patch_embed.*` | `patch_embed.temporal_conv.*` |
| `student.blocks.*.mlp.fc1/fc2.*` | `blocks.*.mlp.0/2.*` |
| `student.norm.*` | `fc_norm.*` for mean pooling |

Other student parameter names retain their suffix after removing `student.`.
The mask token and reconstruction head are specific to pretraining and excluded;
non-student heads and optimizer state are not used.

The original checkpoint has a `(1, 129, 200)` channel-position table and a
`(1, 16, 200)` temporal table. Keep all channel embeddings, selecting the 64 EEG
channels by name in each forward pass. For 800 samples, Braindecode creates a
`(1, 5, 200)` temporal table; copy its first five entries from the checkpoint.
Its forward pass uses the first four for the four one-second patches. Loading
the unsliced temporal table would still cause a shape mismatch, even locally.
No interpolation is needed for this conversion.

The backbone uses mean pooling to match the original fine-tuning default.
Its `fc_norm` receives the pretrained final normalization weights. Every
converted backbone key and shape must match (`strict=True`); only afterward do
we add the randomly initialized single-logit head with the original small
initialization scale. Tests compare every loaded backbone tensor directly with
the corresponding original student tensor, allowing only the documented slice.

The check runs in evaluation mode for deterministic layer behavior, computes
BCE loss, and calls backward to verify finite gradients throughout the model.
Evaluation mode does not disable autograd. There is no optimizer step and no
weights are updated or saved. All backbone parameters remain trainable for the
subsequent fine-tuning lesson.

Default device is CPU. CUDA and MPS can be requested with `--device`, but must
be available in your environment. Only CPU has been verified in this lesson.
The pretrained source is now the local original checkpoint. Exact prediction
equivalence to the old framework would still require comparing forward implementations.
The next step is EEGClassifier with explicit validation data and checkpoint
selection, followed by reproducing the original optimization settings.

## Lesson 4: EEGClassifier training preview

Run this in the Braindecode environment:

```bash
python src/labram_classification_lessons/train_preview.py --contrast solo-coordination --fold 0
```

This script reuses the data and local-checkpoint functions from earlier lessons.
You do not need to run those scripts first. It selects one training dyad and one
validation dyad according to the full fold manifest, then takes the first four
windows from each participant/condition file. Defaults are 16 training windows,
16 validation windows, two epochs, and batch size four. No test EEG is loaded.
This deliberately small, balanced subset is a smoke test, not an estimate of
performance on unseen participants. The full validation fold is not used yet.

### The training interface

The central code is `make_classifier`, followed by `net.fit(training, y=None)`.
The dataset already supplies labels, so no separate y array is needed.

- `module=model`: the already loaded pretrained network; no `module__` overrides
  are passed, so initialization preserves its pretrained parameters.
- `criterion=BinaryLoss`: BCEWithLogitsLoss with a target-dtype conversion. The
  installed EEGClassifier loader casts class labels to int64, so changing the
  dataset's label dtype alone would not be sufficient.
- `optimizer=torch.optim.AdamW`: the update rule used after backpropagation.
- `train_split=predefined_split(validation)`: use our explicit held-out dataset;
  never let the classifier randomly carve validation windows out of training.
- `cropped=False`: one prediction per four-second window. Overlap between
  windows does not require Braindecode's dense/cropped-prediction mode.
- `iterator_train__drop_last=False`: retain the final partial batch.
- `EpochScoring`: calculate validation balanced accuracy each epoch.
- `Checkpoint`: save weights when that score strictly improves, then restore
  the best weights at training end. Ties retain the earliest best epoch.

Each epoch comprises training batches (forward, loss, backward, optimizer step)
followed by validation batches (forward and scoring, without weight updates).
The model's backbone and classification head both train. Class probabilities
returned by `predict_proba` have two columns: P(class 0), P(class 1). The loss
continues to receive raw, single-output logits.

This introductory run uses a constant learning rate of 1e-4 and uniform AdamW
weight decay of 0.05. It intentionally does not yet reproduce the original
warmup/cosine schedule, layer-wise learning-rate decay, parameter-specific
weight decay, gradient accumulation, or DropPath settings. Those belong to the
full-experiment lesson. Do not compare preview scores to the manuscript.

### Saved outputs

Every run creates a fresh git-ignored directory under this folder's `runs/`.
An explicit `--output-dir PATH` must also be new, to avoid overwriting a run.

- `best_model.pt`: validation-selected model weights, including the wrapper's
  backbone and trained head. This is not an optimizer-resume checkpoint.
- `history.json`: complete epoch history, including losses and validation scores.
- `config.json`: contrast, label mapping, channel order, checkpoint path, seed,
  and training settings.
- `fold_manifest.csv`: the full file-level assignment, including the untouched
  test fold. Listing test paths does not load or evaluate the test signals.
- `train_windows.csv` and `valid_windows.csv`: the exact preview selections.
- `validation_predictions.csv`: metadata and predictions from the restored model.

For example, you can adjust `--epochs 3 --batch-size 4`, or select CUDA on the
HPC with `--device cuda`. This command remains a preview regardless of epoch
count; there is no full-experiment flag yet.

The tests verify actual optimizer updates, explicit validation selection, use
of the partial final batch, two-column binary probabilities, and restoration
of the earliest best checkpoint. The real-data CPU preview also completed two
epochs and restored epoch 1; its validation balanced accuracy was 0.5.

Next: reproduce the full optimization protocol, train a complete fold, and
only then evaluate its held-out test dyads using the validation-selected model.

API references:
- [EEGClassifier](https://braindecode.org/dev/generated/braindecode.classifier.EEGClassifier.html)
- [predefined_split](https://skorch.readthedocs.io/en/stable/helper.html#skorch.helper.predefined_split)
- [Scoring and checkpoint callbacks](https://skorch.readthedocs.io/en/stable/callbacks.html)
