# Project 2: paired Circling data

Stage 0 provides an audit and paired cache for the **local dataset used by Project 1**. Stage 1 diagnostics and Stage 2 single-person SSL are available below.
Use the existing `braindecode-env` environment. No new dependencies are required.

The user-confirmed HPC source is `/work3/s204684/Thesis/circling_eeg/`. It is not mounted on the development Mac;
the real-data audit and cache generation must run on the HPC. No real-data results have been produced here.

## HPC submission

After syncing this module to the existing HPC checkout, submit the CPU-only audit from the repository root:

```bash
mkdir -p outputs errors
bsub < src/hyperscanning_contrastive/hpc/circling_data.lsf
```

The script defaults to the existing project's HPC checkout and `/work3/s204684/Thesis/braindecode-env` environment.
Override `PROJECT_DIR` or `VENV` if those have changed. It writes a new directory under
`/work3/s204684/Thesis/data/processed/circling_contrastive_audit_<job-id>` and prints the exact result path.
Review `audit.json`, `census.csv`, and any alignment failures before generating signals:

```bash
STAGE=prepare bsub < src/hyperscanning_contrastive/hpc/circling_data.lsf
```

Preparation writes a separate `circling_contrastive_prepare_<job-id>` directory. No GPU is requested. This template
has been syntax-checked locally but has not been submitted or runtime-tested on the cluster.

## Run

From the repository root, set `PYTHONPATH=src`. Either supply `--data-dir` or set `CIRCLING_DATASET_PATH` in the
environment/root `.env`. The directory can contain the flat `pair..._ave_epo.fif` files or their `preprocessed/` folder.

```bash
python -m hyperscanning_contrastive audit \
  --data-dir /path/to/local/circling \
  --bad-segments /path/to/local/circling/dict_with_short_bad_epochs.pickle \
  --output reports/circling_contrastive_audit

python -m hyperscanning_contrastive prepare \
  --data-dir /path/to/local/circling \
  --bad-segments /path/to/local/circling/dict_with_short_bad_epochs.pickle \
  --output data/processed/circling_contrastive

python -m pytest tests/hyperscanning_contrastive
```

Each output directory must be new. Failed runs leave an incomplete directory for diagnosis and no completion marker.
Audit reads EEG metadata; prepare additionally validates and saves signals. Only load the trusted local artifact
pickle: Python pickle loading can execute code.

Default conditions are `closed1 open closed2`. If the local files contain only the Project 1 conditions, explicitly
select `--conditions closed1 open`. Missing requested conditions fail instead of silently changing the experiment.

## Artifact and timing contract

- Remove the first 6 epochs (two three-condition practice cycles), or 12 for pair2030, as in Project 1.
- Analyze the half-open interval `[2,20)` seconds relative to each condition onset. This is 18 seconds after cropping.
- Dictionary key `<group>_<condition>` contains condition-local indices of **6-second sections**, not four-second
  windows. Index `i` maps to experimental condition trial `i // 3` and section `i % 3`.
- Relative to the cropped origin, sections are `[0,6)`, `[6,12)`, `[12,18)`. Their original condition times are
  `[2,8)`, `[8,14)`, `[14,20)`.
- A window is excluded if `window_start < bad_end and window_end > bad_start`. Merely touching an endpoint is safe.
- The dictionary is dyad-level: the same clean intervals are used for both participants. Never retain only one member.
- Missing dictionary keys mean no listed exclusions, matching Project 1; every such key is reported for review.
  Out-of-range indices fail. Unused dictionary keys are also reported.

Default `--window-strategy max-clean` subtracts all bad sections, then packs four-second windows left-to-right within
each contiguous clean run. This maximizes their number, because a clean run of length L accommodates `floor(L/4)`
non-overlapping windows. Windows may cross boundaries between **clean** six-second sections.

| Bad sections | Retained windows relative to the cropped origin |
| --- | --- |
| None | 0–4, 4–8, 8–12, 12–16 |
| First | 6–10, 10–14, 14–18 |
| Middle | 0–4, 12–16 |
| Last | 0–4, 4–8, 8–12 |

An optional `--window-strategy fixed-grid` considers starts 0, 4, 8, 12 only and discards intersecting windows. It is
useful for a timing sensitivity comparison but can retain fewer windows. Neither strategy uses overlapping windows.
Max-clean offsets depend on exclusions: later retrieval must match exact offsets and account for available trials.
Do not let missing-window patterns or differing candidate counts become a retrieval shortcut. Windows ending at
20 seconds have no end-transition buffer; report/compare them separately if transition responses are a concern.

## Pairing checks

Exactly two distinct participant files must exist per `pair<digits>` group. The tool requires identical original
event arrays, event-name mappings, epoch selections, sampling rates, and time axes across participants. Original
selection IDs and event samples must strictly increase. Selection IDs may start above zero or contain gaps.
The dataset owner supplied these FIF files together with the six-second artifact dictionary; the user confirmed
that the dictionary's exclusions have not been applied to these files. Treat the supplied FIF row order as the
reference, matching Project 1: remove practice rows, select the condition, then interpret dictionary index `i`
as condition-local trial `i // 3`, section `i % 3`. Do not drop additional full trials before this mapping.
Historical `drop_log` entries (including `USER` and `BAD boundary`) do not invalidate this source contract or
insert missing trials into the dictionary indexing. Selection IDs remain provenance, not artifact indices.
Pairing still requires matching event arrays and selections, rather than relying on row counts alone.

Practice counts, full analysis-interval coverage, EEG channel types, and dictionary bounds are checked. Local group
and participant names are preserved. Array member 0/1 follows sorted filenames and conveys no social role.

## Outputs

- `files.csv`: participant/source provenance, sizes, modification times, source hashes for prepared caches, group folds.
- `trials.csv`: every requested experimental condition trial, including those with zero retained windows and exclusions.
- `windows.csv`: one row per retained **pair**, original event/epoch IDs, source sample slices, exact temporal offsets,
  participant IDs, array index, and group fold.
- `census.csv`: counts by group and condition, including fully excluded conditions.
- `audit.json`: completion/provenance marker for metadata-only audits.
- `pair_<group>.npy` and `cache.json`: prepare-only paired signals and successful cache completion marker.

Each array has shape `(paired_windows, 2, 64, 800)`, float32 **microvolts**, at 200 Hz. Both participants use the same
physical interval and the Project 1 channel order. Resampling occurs after extraction, per window, as in Project 1.
No extra rereference or baseline correction is added beyond MNE's loading of the source's stored preprocessing/projectors.
Future model loaders must divide by 100 exactly once, as in Project 1. Metadata-only audit does not establish signal
finiteness; prepare checks every retained signal. Prepared arrays and manifests are SHA256-hashed.

Group fold assignment reproduces Project 1's sorted IDs, NumPy RandomState(42) shuffle, and ten consecutive KFold
partitions when at least ten groups exist. Smaller synthetic audits get fewer folds and are not evaluation-ready.
Even a ten-fold cache does not commit us to a training/validation/test protocol; freeze that protocol before training.
`evaluation_ready` only checks minimum fold count and retained coverage, not statistical power or final gallery size.

See [PLAN.md](PLAN.md) for the research roadmap. The public release was researched as background; its format and
artifact-marker description are not applied to this local cache.

## Stage 1: synthetic engineering controls

Run the CPU controls in a new output directory:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive controls --output runs/contrastive-controls
```

This needs Torch but no cache, pretrained checkpoint, or GPU. It checks identical-view retrieval, tiny-set
contrastive fitting, generalization to held-out synthetic shared latent vectors mixed through different sensors,
and retrieval of independent random vectors. The synthetic experiment uses two linear adapters to isolate
loss/optimizer behavior; these are not the proposed LaBraM architecture. `controls.json` records the seed,
losses, ranks, chance rate, and explicit engineering gates. A failed gate returns a nonzero exit code.
The independent control averages 100 synthetic repeats; its tolerance is an engineering check, not a scientific
significance test. Tied similarities receive fractional top-one credit, so collapsed embeddings score at chance.

The loss supports an explicit candidate mask and symmetric InfoNCE. Every anchor needs its diagonal positive
and at least one eligible negative in both directions. General real-EEG candidate construction and
trial-level permutation nulls remain outstanding. LaBraM forward/backward checks and real-cache tiny-set overfitting
are now available through the `overfit` command below. Synthetic results do not establish hyperscanning performance.

## Stage 1: tiny-set LaBraM EEG overfit

Use the completed paired cache (`cache.json` must exist). The default experiment freezes pretrained LaBraM,
extracts pooled features once, and fits a shared 200→256→128 projection to eight partner pairs. This can run on CPU:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive overfit \
  --cache /path/to/completed/cache --output runs/circling-tiny-pretrained
```

For the HPC wrapper, create `outputs/` and `errors/`, then submit:

```bash
CACHE=/path/to/completed/cache bsub < src/hyperscanning_contrastive/hpc/circling_overfit.lsf
```

The wrapper lives in the existing git-ignored `hpc/` directory; copy it explicitly when syncing through Git.
Use `--initialization random` for the architecture control: no pretrained file is read or hashed. For encoder
fine-tuning, run the Python command with `--train-backbone --device cuda` inside a GPU allocation and a new output
directory. The CPU wrapper deliberately performs only the frozen-backbone diagnostic.

The loader verifies manifest hashes and the selected array hash, channel order, microvolt units, shape, and sampling
contract. Input scaling is `/100` exactly once inside the encoder. It reuses Project 1's native LaBraM checkpoint
conversion without modifying the classification package; the binary head is replaced by identity before the shared
projection. Both branches use the same model. Dropout/stochastic depth are disabled for deterministic tiny-set fitting.

Selection is deterministic: development fold 0, the first eligible dyad, condition `open`, offset zero, and one window
per distinct trial. Override with `--development-fold`, `--group`, `--condition`, and `--pairs` when needed. A requested
group must belong to the requested fold. Keep these development choices fixed; do not cycle through held-out groups
to get a passing result. All candidates share participant identity, condition, and offset. No trial repeats appear as
false negatives. This is deliberately a small memorization task, not held-out partner retrieval or Stage 2 SSL.

Outputs are `selected_windows.csv`, `history.csv`, `model.pt`, and `overfit.json`, including baseline/final bidirectional
retrieval, loss, chance, cache/checkpoint fingerprints, and settings. Pass requires top-one ≥0.99 and final loss below
10% of initial loss. Failure returns a nonzero exit code but retains the report; inspect training curves before changing
settings. Frozen features can limit fitting, so failure alone does not invalidate the encoder or the scientific idea.

After this check, run Stage 2 single-person augmented-view learning with held-out-group evaluation below.
Real-EEG candidate masks and trial-permutation nulls are still needed before the main partner-retrieval experiments.

## Stage 2: single-person augmented-view SSL

This pilot asks whether two independently augmented versions of a person's EEG window can be matched on unseen
dyads. It does **not** use the partner as a positive and does not test interactive versus non-interactive coupling.

The default fixed split is validation fold 1, test fold 2, and all remaining folds for training. Fold 0, used in
Stage 1 development, remains in training. For the 52-dyad cache this gives 41 training, 6 validation, and 5 test dyads.
This is one exploratory split, not ten-fold CV. All three conditions and both participants are used. Do not change
folds between initialization comparisons or move validation/test groups into training after inspecting their results.

The first pilot deliberately uses only clean windows at crop offset zero (source epoch seconds 2–6), one per trial.
Each gallery has eight different trials from one participant, dyad, and condition. This removes participant identity,
condition, and window offset as ways to identify a positive among its candidates. Training samples these strata
uniformly and chooses fresh trial subsets. Validation and test galleries are saved once using signal-independent
random selection. Both initializations use the same galleries and augmentation draws. An insufficient stratum fails
explicitly instead of silently dropping a dyad. General multi-offset gallery masking is future work.

Two independent views receive per-window amplitude scaling in [0.9, 1.1], independent Gaussian noise at 2% of each
channel's RMS, and independent channel dropout with probability 0.05. Time samples and channel positions never move.
Use `--amplitude`, `--noise`, and `--channel-drop` to disable individual transformations for development ablations;
these settings are saved. Augmentation is performed in microvolts, followed by the adapter's single `/100` scaling.

### Run the matched pretrained/random pilots

```bash
CACHE=/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487 \
  INITIALIZATION=pretrained bsub < src/hyperscanning_contrastive/hpc/circling_ssl.lsf

CACHE=/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487 \
  INITIALIZATION=random bsub < src/hyperscanning_contrastive/hpc/circling_ssl.lsf
```

Copy the wrapper explicitly when syncing: the existing `hpc/` directory is Git-ignored. It uses the same V100 queue
and host selector as Project 1 and creates a separate run directory keyed by initialization and job ID. Queue runtime
is a request, not a measured completion estimate. Direct invocation inside a GPU allocation is:

```bash
PYTHONPATH=src python -m hyperscanning_contrastive ssl \
  --cache /path/to/cache --output runs/ssl-pretrained --device cuda \
  --initialization pretrained --epochs 20 --steps-per-epoch 100 --pairs 8
```

Unlike `overfit`, **Stage 2 trains the backbone by default**. Use `--freeze-backbone` for a projection-only control.
Pretrained backbone LR is 1e-5, random backbone LR is 1e-4, projection LR is 1e-3; both use AdamW weight decay 0.01,
gradient clipping at 1, temperature 0.1, and the same 2,000-update budget. Different backbone LRs are explicit pilot
settings, not a claim of optimal or strictly hyperparameter-identical initialization comparison. Random runs never
read or hash the pretrained checkpoint. Inputs/manifests are verified before training; arrays are memory-mapped.

### Selection and interpretation

`history.csv` reports training loss and fixed-gallery validation metrics. Save and compare `validation_initial.csv`
(untrained projection), `validation_initial_pooled.csv` (raw pooled features), and `validation_best.csv`. Selection
uses **validation MRR**, averaging members, then conditions, then dyads equally. The initial model is eligible as
epoch 0; ties keep the earlier checkpoint. No test metric is computed during training. The fixed epoch budget has
no test-driven early stopping. `ssl.json` records the selected epoch, baseline metrics, nominal chance, and checkpoint,
settings, and gallery hashes. `settings.json` contains the complete pilot configuration; `splits.csv` records dyads.

For eight candidates, top-one chance is 0.125 and MRR chance is approximately 0.340. Ties receive the expected metric
under random tie-breaking, so collapse cannot exploit candidate ordering. Per-gallery embedding standard deviation
is included as a collapse diagnostic. Strong initial retrieval may make this task too easy: high final accuracy alone
does not establish a benefit from SSL. Compare the initial/best results and per-dyad values; a best epoch of zero is
an informative lack of validation improvement. Same-window view matching is an engineering/representation test,
not evidence of cross-person information. It also cannot establish physiological validity of an augmentation.

### Explicit test evaluation after locking the pilot

Once both configurations are fixed, evaluate each selected run with the same saved galleries:

```bash
PYTHONPATH=src python -m hyperscanning_contrastive ssl-evaluate \
  --cache /path/to/cache --run /path/to/completed/ssl-run \
  --output runs/ssl-pretrained-test --device cuda
```

This verifies the original cache, settings, gallery, and model hashes; evaluates initial pooled features, initial
projection, and the validation-selected model; and writes per-gallery CSVs plus `test.json`. Use a new output
directory. Test results must not become a signal for repeated tuning. Report dyad-level variability, not window-level
replicates as independent evidence. Multiple seeds and uncertainty estimates remain necessary for research claims.

Stage 3 partner alignment is available separately below; these Stage 2 commands do not perform it.

## Stage 2: validation correspondence control

When same-window retrieval saturates even before training, use this engineering check before increasing task
complexity. It requires no retraining and never scores test signals:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive ssl-correspondence \
  --cache /path/to/cache --run /path/to/completed/ssl-run \
  --output runs/ssl-validation-correspondence --permutations 200 --device cpu
```

Run it once for each pretrained/random run. On HPC, set `CACHE` and `RUN` and submit
`bsub < src/hyperscanning_contrastive/hpc/circling_correspondence.lsf`. Copy this Git-ignored wrapper explicitly.

The command verifies saved settings, checkpoints, cache, and validation gallery hashes. It evaluates initial pooled
features, the initial projection, and the selected checkpoint using the same augmentation seed as validation. The
selected checkpoint can be epoch zero, especially when the initial model already has perfect validation MRR.
Embeddings are computed once per gallery/representation; permutations reuse them on CPU.

Each uniform permutation reassigns the second view to designated diagonal positives within the same participant,
condition, and offset gallery, consistently in both retrieval directions. Fixed points are allowed: forcing all
matches to be wrong would change the expected chance score. The same random permutations are replayed across
representations and runs when galleries match. The expected shuffled means for eight candidates are top-one 0.125
and MRR 0.339732. Finite-repeat estimates fluctuate around these values; no exact equality gate is imposed.

`correspondence.json` reports real scores, shuffled means, between-repeat SD, and Monte Carlo SE of the shuffled
means. These SEs describe simulation precision, not uncertainty across people. Per-gallery and per-repeat macro
CSVs preserve equal member/condition/dyad weighting. This is a label-assignment engineering control, not a
physiological surrogate or a hypothesis test of hyperscanning; no scientific p-value is produced. A high real score
and chance shuffled score validate correspondence sensitivity, not benefit from SSL or pretraining.

After this check, keep test data untouched while deciding whether a harder Stage 2 task (such as disjoint sensor
views) is needed. Do not tune the learning rate solely to address a saturated ranking metric.

## Stage 3: frozen partner-retrieval baseline

The `partner-baseline` command queries one participant's EEG against the **other participant's EEG** from eight
trials of the same condition, with simultaneous trial as the positive. It evaluates both directions on validation
fold 1 only, using source seconds 2–6 (crop offset zero). It uses original frozen pooled LaBraM features and cosine
similarity, without augmentation, projection, SSL adaptation, or training. Test fold 2 remains unscored.

```bash
export CACHE=/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487
INITIALIZATION=pretrained bsub < src/hyperscanning_contrastive/hpc/circling_partner_baseline.lsf
INITIALIZATION=random bsub < src/hyperscanning_contrastive/hpc/circling_partner_baseline.lsf
```

Copy the Git-ignored HPC wrapper explicitly. This is a CPU job and requires no Stage 2 run path. Direct invocation:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive partner-baseline \
  --cache /path/to/cache --output runs/partner-pretrained --initialization pretrained --device cpu
```

The baseline fixes validation/test folds to 1/2. `--pairs`, `--seed`, `--permutations`, and `--permutation-seed` configure
candidate count and reproducibility; use identical values across initializations. Random initialization never reads
or hashes a pretrained checkpoint. The original pretrained model is loaded independently of Stage 2 selection.

Gallery selection reuses the Stage 2 member-0 validation subsets, then reads **both members at those exact trial
rows**. It does not independently sample the partner's trials. All participants/conditions in validation are included;
the cache checks its input contract and hashes. `galleries.csv` must have the same SHA256 across initialization runs.
There is no pooled gallery spanning different target identities or conditions.

`partner.json` contains true and shuffled metrics separately for `closed1`, `open`, and `closed2`, with equal dyad
weighting. It records sample SD across dyads separately from Monte Carlo uncertainty of the shuffled mean.
`directions.csv` retains A→B and B→A; `queries.csv` retains individual query ranks; `dyads.csv` averages directions.
`similarities.npz` holds the actual score matrices in the saved gallery order. `shuffled_dyads.csv` records all
randomization repeats. `open_minus_closed_by_dyad.csv` records within-dyad open minus mean(closed1, closed2), and the
report summarizes those paired differences. This contrast is descriptive and does not replace inspecting both closed
conditions separately.

Uniform permutations reassign whole trial correspondence within each dyad/condition gallery, consistently in both
directions; fixed points remain allowed. The same assignments are replayed across initializations. With eight
candidates, top-one chance is 0.125 and MRR chance is 0.339732. These are exploratory correspondence controls:
unrestricted trial exchangeability may fail under slow drift/block structure, and this first baseline does not adjust
for shared cues or movement. No scientific significance claim or physiological interpretation follows from this check
alone. Multiple seeds, stronger matched controls, and a locked held-out evaluation remain later work.

The learned shared-projection step is available through `partner-train` below. Limited encoder adaptation remains
future work; `partner-baseline` itself performs no training.

## Stage 3: learned shared projection with shuffled-training control

`partner-train` extracts original frozen LaBraM features once, then fits the same shared 200→256→128 head to both
partners. Both experimental arms run automatically: true correspondence and a fixed shuffled-training-pair control.
No Stage 2 checkpoint is used. No backbone weights are updated. No signal augmentation is applied.

```bash
export CACHE=/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487
INITIALIZATION=pretrained bsub < src/hyperscanning_contrastive/hpc/circling_partner_train.lsf
INITIALIZATION=random bsub < src/hyperscanning_contrastive/hpc/circling_partner_train.lsf
```

Copy the Git-ignored wrapper explicitly when syncing. The GPU is used for bounded-batch feature extraction; both small
heads train on CPU with detached features. The direct command is `PYTHONPATH=src python -m hyperscanning_contrastive
partner-train --cache <cache> --output <new-run> --initialization pretrained --device cuda` (on one line).
Random initialization never reads or hashes pretrained weights. No extra jobs are needed for shuffled-training arms.

The split is fixed: validation fold 1, untouched test fold 2, remaining folds training. Use one clean offset-zero
window per available trial, all three conditions and both partners. Training uniformly samples dyad/condition strata,
then eight distinct A trials. True-pair positives use the same B trials. The shuffled control maps each A trial to a
fixed randomly permuted B trial within that training dyad/condition, then selects candidates using that mapping.
This is a bijection over the entire available stratum, not fresh noise each update. Fixed points are allowed and are
recorded in `shuffled_training_pairs.csv`. It may learn arbitrary training correspondences; that is the intended
control. It is not a physiological surrogate or a formal null-distribution experiment.

Both arms start from identical projection weights and use identical stratum/A-trial sampling schedules. Each uses
AdamW LR 1e-3, weight decay 0.01, temperature 0.1, gradient clipping at 1, and 20 epochs of 100 updates. Validation
always evaluates **true** partner correspondence on the same galleries as `partner-baseline`. Compare gallery hashes
across runs. Selection uses equal-condition, dyad-macro validation MRR, includes the untrained projection as epoch 0,
and retains the earlier checkpoint on ties. It never selects for an interactive/non-interactive gap.

Outputs:

- `alignment.json`: raw pooled and untrained-projection baselines, selected results for each arm, per-condition scores,
  settings and provenance. Test signals are never extracted or scored.
- `true/` and `shuffled/`: `history.csv`, per-epoch validation query CSVs, `validation_best.csv`, `dyads.csv`, and
  `best_projection.pt`.
- `backbone.pt`, `initial_projection.pt`, `galleries.csv`, `splits.csv`, and `shuffled_training_pairs.csv` preserve the
  frozen encoder, shared starting point, and data mappings.

A useful result would be a true-pair projection improving over raw pooled features, the untrained projection, and
the shuffled-training control on validation dyads. Similar improvements in both arms suggest generic feature
reweighting or selection effects rather than a benefit of true correspondence. With six validation dyads, compare
per-dyad patterns and keep results exploratory. Failure does not prove absence of shared EEG information. Limited
backbone adaptation and locked held-out evaluation remain later steps; these training commands do neither.

### Alignment learning diagnostics

`partner-train` now records diagnostics automatically; use the same command/HPC wrapper in a new output directory.
Existing runs cannot provide historical optimizer updates retrospectively, so rerun to collect these measurements.
The learning rate, batch schedule, arm mapping, and validation-MRR checkpoint rule remain unchanged.

Each arm's `history.csv` adds fixed-gallery `train_true_*`, `train_objective_*`, and `validation_*` metrics. Training
truth always uses synchronous partners; the objective evaluation uses the arm's actual training mapping (fixed
shuffled pairs in the control). Fixed training galleries are saved in `training_galleries.csv` and never change
between epochs or arms. These are training-set diagnostics, not extra independent validation data.

For each gallery the diagnostics report cosine positive/negative means and their margin, loss, retrieval metrics,
mean embedding norm, normalized coordinate standard deviation, and effective rank. Geometry is computed within
each participant and averaged, so a difference between participants cannot alone make a collapsed representation
appear diverse. Effective rank uses entropy of singular values of centered L2-normalized embeddings; its maximum
is min(K−1, feature dimension), hence seven for eight candidates. Exactly constant embeddings return zero. Rank
alone can hide tiny variation, so interpret it together with normalized spread and cosine margins.

`gradient_norm_mean` is the epoch mean of pre-clipping gradient norms. `update_norm_mean` is the mean actual
parameter-step L2 norm, including AdamW effects; `relative_update_mean` divides each update by the pre-step parameter
norm. `diagnostic_galleries.csv` retains per-dyad/condition measurements each epoch. `diagnostics_version: 1` in
`alignment.json` identifies instrumented runs. Test signals remain unused.

Interpretation: improving fixed training scores with flat validation suggests poor transfer/overfitting; both flat
with tiny margins suggests ineffective discrimination; collapsing normalized spread/rank suggests collapse; tiny
or absent optimizer updates suggests an optimization problem. None of these observations alone proves absence of
shared EEG information. Inspect both the true and shuffled arms before selecting the next intervention.

### Geometry ablation: training-centered, bias-free linear head

Use `--projection-profile centered-linear` (or `PROJECTION_PROFILE=centered-linear` with the existing HPC wrapper)
to test a simpler geometry when positive and negative cosines are both near one despite nonzero optimizer updates.
The default `legacy-mlp` preserves the previous training architecture and settings.

```bash
export CACHE=/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487
PROJECTION_PROFILE=centered-linear INITIALIZATION=pretrained \
  bsub < src/hyperscanning_contrastive/hpc/circling_partner_train.lsf
PROJECTION_PROFILE=centered-linear INITIALIZATION=random \
  bsub < src/hyperscanning_contrastive/hpc/circling_partner_train.lsf
```

Sync the updated Git-ignored wrapper as well as Python code. This profile subtracts a single 200-dimensional global
mean fitted using all extracted training windows from both participants, weighted equally per window. Neither
validation nor test contributes. The frozen mean is shared across participants, conditions, and arms; it is not fitted
per batch, gallery, subject, or held-out dyad. The subsequent trainable head is a bias-free 200→128 linear map.
Cosine normalization remains inside the scoring function. No per-feature standardization or whitening is applied.

The run reports the original raw pooled baseline, a new centered-pooled baseline, the untrained profile-specific
head, and the selected true/shuffled heads. `validation_centered_pooled.csv` isolates the effect of subtracting the
mean without training. `feature_center.pt` stores the mean, which is also a buffer in the profile's saved projection
checkpoints. `projection_profile` and `feature_center_sha256` identify the variant in `alignment.json`.

This is a geometry ablation, not a proven collapse remedy: centering and head architecture both change, so any
benefit cannot automatically be assigned to one of them. The extra centered-pooled baseline helps separate the
centering effect. LR, temperature, epoch budget, true/shuffled mapping, galleries, diagnostics, and validation-MRR
selection remain unchanged. Test dyads remain untouched. Compare normalized spread and cosine margins as well as
training/validation MRR, and require a true-pair benefit over the shuffled-training control before interpreting an
improvement as useful correspondence learning.

### Stage 3: limited backbone adaptation

`partner-adapt` tests whether updating the representation helps when the frozen shared head does not
consistently outperform shuffled training. It starts from the original pretrained checkpoint and original
shared MLP, without centering or Stage 2 SSL weights. Defaults train the last **two transformer blocks**,
final pooling normalization, and projection. Encoder LR is `1e-5`; projection LR remains `1e-3`.
AdamW weight decay `0.01`, temperature `0.1`, 20 epochs × 100 steps, eight pairs, seeds, validation galleries,
and fixed within-dyad/condition shuffled mappings match the frozen experiment. Both arms run in one job,
starting from identical full model weights with the same minibatch sampling schedule.

This is a limited pilot, with constant learning rates and no warm-up, rather than the larger schedule
search in the plan. Evaluation mode disables stochastic layers during fitting but gradients remain enabled.
No augmentations are applied. Both encoder and projection train on the requested device.
Validation fold 1 selects each arm's checkpoint by macro MRR (including epoch zero); fold 2 is not evaluated.
For a clean comparison, use the earlier **legacy-mlp** frozen run, since centered-linear changes the head too.

From the HPC repository root, after synchronizing the new code and wrapper:

```bash
mkdir -p outputs errors
CACHE=/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487 \
    bsub < src/hyperscanning_contrastive/hpc/circling_partner_adapt.lsf
```

The `hpc/` directory is ignored by Git in this repository; copy the wrapper explicitly if deploying via Git.
The job writes `runs/circling_adaptation_pretrained_<job-id>/`. Alternatively, inside a GPU allocation:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive partner-adapt \
    --cache /work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487 \
    --output runs/circling_adaptation_pretrained_manual --device cuda \
    --initialization pretrained --unfreeze-blocks 2 --backbone-lr 1e-5 \
    --pairs 8 --epochs 20 --steps-per-epoch 100 --seed 42 --permutation-seed 1042
```

Inspect `adaptation.json`, both `history.csv` files, and both `dyads.csv` files. Full initial/best/last
checkpoints are saved. Epoch histories include true and assigned-objective training retrieval, validation
retrieval, embedding geometry, and gradient norms. Training diagnostics use the first four training dyads
(sorted IDs, all conditions) to bound repeated EEG inference; these diagnostic means are **not** directly
comparable to the previous all-training-dyad means. Validation uses all the same fixed galleries.
The report records diagnostic dyad IDs, all trainable parameter names, and provenance hashes.

A useful outcome requires a true-pair validation advantage over both frozen training and the matched
shuffled arm, with per-dyad/condition inspection. Training improvements alone show fitting, not transfer.
A single selected validation peak is exploratory; repeated-seed validation is needed before claiming a
reproducible advantage. Weak results here still leave temporal pooling, signal alignment, and simpler
spectral/behavioral comparators as alternatives; they do not establish absence of shared EEG information.

### Plot partner-training results

Generate simple unsmoothed line plots from a completed `partner-train` or `partner-adapt` run:

```bash
PYTHONPATH=src python -m hyperscanning_contrastive.plotting \
    --run runs/circling_adaptation_pretrained_29770724 \
    --output reports/circling_adaptation_pretrained_29770724
```

This CPU-only command reads the report and the two history CSVs; it does not load EEG or checkpoints.
It accepts `shuffled/history_shuffled.csv` when the original history was renamed. Matplotlib is required.
`retrieval.png` compares validation MRR/top-1 and fixed training-gallery real/assigned-match MRR.
Stars mark the MRR-selected checkpoint in each arm. `diagnostics.png` shows minibatch loss, validation
cosine margin, normalized embedding spread, and within-gallery effective rank. Both figures also export
as vector PDFs. Chance lines are reference expectations, not significance thresholds; no smoothing or
confidence intervals are added. Training diagnostic dyads may differ from the full training population.

### Stage 3: held-out trials from familiar dyads

Use `partner-adapt --validation-protocol within-dyad` for the intermediate generalization experiment.
It retains the original 41 training dyads and randomly holds out eight eligible whole trials per dyad
and condition, with split seed 2042. Both participants inherit the same trial assignment. Only the
first clean offset-zero window (source seconds 2–6) is used, as in the previous pilot. Later windows
are not read. The six original external-validation dyads (fold 1) and five test dyads (fold 2) are
excluded from all model forward passes in this mode. Cache integrity checks still verify all files.

There is no participant-held-out claim: this measures unseen-trial retrieval within familiar people
and sessions. A randomized trial split can benefit from proximity in time; temporally blocked validation
is a later control. Candidate galleries still contain trials from one partner and condition.

The eight held-out trials supply the entire validation gallery in the default configuration. The
remaining 10–12 trials per condition supply training batches of eight. Insufficient strata fail with an
explicit error instead of silently changing the gallery size or excluding dyads. The paired-trial
manifest `trial_splits.csv` records every eligible trial's assignment. Shuffled assignments are formed
only among training trials. Training and validation feature stores remain separate even for the same dyad.

Start from the original pretrained checkpoint and preserve the previous 20 × 100 update budget,
last two trainable blocks plus pooling normalization, original MLP, and learning rates. Both arms run
from identical model weights and matched batch schedules. Checkpoint selection uses within-dyad
validation macro MRR, including epoch zero. Training diagnostics still use four training dyads;
validation now covers all 41 familiar dyads, so its variability is not directly comparable to the
previous six-dyad validation estimate. The report includes split counts and the trial-manifest hash.

After syncing the code and explicitly copying the ignored HPC wrapper:

```bash
mkdir -p outputs errors
CACHE=/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487 \
    bsub < src/hyperscanning_contrastive/hpc/circling_partner_within_dyad.lsf
```

The wrapper writes `runs/circling_within_dyad_pretrained_<job-id>/` and then generates PNG/PDF figures
under `plots/` using Matplotlib. Review `adaptation.json`, both histories, `trial_splits.csv`, and the plots.
No cache rebuild is required. The original unseen-dyad mode remains the command's default.

For an existing GPU allocation, use:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive partner-adapt \
    --cache /work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487 \
    --output runs/circling_within_dyad_pretrained_manual --device cuda \
    --initialization pretrained --pairs 8 --epochs 20 --steps-per-epoch 100 \
    --unfreeze-blocks 2 --backbone-lr 1e-5 --seed 42 --permutation-seed 1042 \
    --validation-protocol within-dyad --validation-trials 8 --split-seed 2042
```

### Spectral comparator on the same within-dyad experiment

`partner-spectral` reads a completed within-dyad adaptation run and reconstructs its exact trial split,
validation galleries, training diagnostic galleries, and shuffled training assignments. It checks their
SHA256 hashes before fitting, and requires the same prepared cache. It never reads model checkpoints.
The reference run supplies the seeds, gallery size, and training schedule; CLI training overrides are
not used by this comparator. Keep its manifests alongside `adaptation.json`.

Features reuse Project 1's fixed spectral extractor without changing it: per-channel log10 absolute
power in delta (1–4), theta (4–8), alpha (8–13), beta (13–30), and low gamma (30–45 Hz). Welch spectra
use two-second periodic Hann windows, one-second overlap, and constant detrending at 200 Hz. Band
power uses trapezoidal integration including band endpoints and a 1e-12 power floor, in microvolts
squared. The original four-second input yields 320 features; no LaBraM scaling by 100 is applied.

One global feature mean/SD is fitted on training windows from both participants only, then reused
unchanged. The direct baseline retrieves with cosine similarity of these standardized features.
The learned comparator adds a shared bias-free 320→64 linear map (20,480 parameters) followed by
cosine-based symmetric contrastive loss. True and fixed-shuffled arms have identical initial weights
and sampling schedules, AdamW LR 1e-3, weight decay .01, temperature .1, and gradient clipping at 1.
Validation MRR selects checkpoints, including epoch zero. All extraction and fitting run on CPU.
The six external-validation and five test dyads are not used for feature extraction or fitting.

After syncing code and copying the ignored HPC wrapper, submit from the repository root:

```bash
mkdir -p outputs errors
CACHE=/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487 \
REFERENCE_RUN="$PWD/runs/circling_within_dyad_pretrained_29771026" \
    bsub < src/hyperscanning_contrastive/hpc/circling_partner_spectral.lsf
```

Or run directly in the environment, without a GPU:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive partner-spectral \
    --cache /work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487 \
    --run runs/circling_within_dyad_pretrained_29771026 \
    --output runs/circling_spectral_manual
PYTHONPATH=src python -m hyperscanning_contrastive.plotting \
    --run runs/circling_spectral_manual --output runs/circling_spectral_manual/plots
```

Read `spectral.json` (`direct`, `initial_projection`, and both `arms`), the two histories, and `plots/`.
The retrieval plots include the direct spectral baseline. This is a lower-capacity representation
comparator, not a pure feature-only ablation of the original MLP. It does not measure interbrain phase
synchrony or time-resolved coupling. Results remain exploratory validation results.

## Stage 2.5: temporal bridge before partner retrieval

The augmented-window task was almost perfect before training, including with random weights.
It verified correspondence and optimization but did not establish useful learned generalization.
Partner adaptation subsequently improved training retrieval much more than held-out retrieval;
changing temperature alone cannot establish that the desired shared information exists.

The next bounded experiment retrieves a **different non-overlapping window from the same person
and trial**, among other trials from that person and condition. Source interval is 2–6 seconds;
target is 6–10 seconds. Only cached clean windows at crop offsets 0 and 4 are eligible. Existing
symmetric artifact exclusions remain intact. A fresh deterministic split holds out eight whole
paired trials per dyad/condition, after intersecting window availability. Both people share the
trial split. Original fold 1 and fold 2 signals are never forwarded to the encoder. This is a new
experiment: its galleries must not be compared numerically as if identical to earlier partner runs.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive temporal-bridge \
    --cache <paired-cache> --output <new-run> --device cuda \
    --temperatures 0.05 0.1 0.2 --groups-per-update 4
```

Use `hpc/circling_temporal_bridge.lsf` on HPC. These wrappers may be ignored by Git: copy the new
wrapper explicitly when deploying. Feature extraction uses frozen original LaBraM once. Head
training uses CPU, global training-only coordinate standardization and a bias-free linear 64-D
projection. Random initialization never reads pretrained weights. No previous SSL checkpoint is
assumed to improve this task.

Each epoch visits **every eligible training trial for both people**. The denominator contains all
remaining trials from one person/condition, typically up to 12, not trials from other participants.
Four such losses are averaged per optimizer step. Thus approximately 40–48 pair presentations per
update do **not** mean 40–48 negatives per query. The report records actual counts. A stratum with
fewer than 16 eligible trials fails explicitly before model loading; do not silently drop dyads.

The three temperatures run true and fixed shuffled correspondence arms from identical initial
heads and sampling orders. Shuffles stay within training person/condition and permit fixed points.
Validation always uses true correspondence. Training diagnostic galleries use eight candidates,
so their MRR is comparable to validation; optimization uses the full training gallery. Temperature
changes loss and gradients, not ranking of fixed cosine scores. The true validation MRR selects
an epoch and temperature; report its matched control, not whichever control is lowest. This is
exploratory selection, not a significance test.

Outputs include `temporal.json`, eligibility census, trial and gallery manifests, shuffled mapping,
per-temperature/arm histories and checkpoints, and automatic `temporal_curves.png`/`.pdf`.
Curves show training-objective and held-out-trial MRR with chance and initial baselines. Raw pooled,
standardized pooled, and initial-projection scores are also saved separately.

Interpret outcomes before adding another model:

| Result | Next decision |
| --- | --- |
| Raw/initial representations already near ceiling | This task is another engineering check, not evidence of learned improvement. Do not tune it further. |
| Training improves but validation does not | Temporal transfer itself is failing; inspect trial drift, features and timing before more partner fine-tuning. |
| True arm improves over initial representations and matched shuffled arm on validation | Repeat the fixed setting with another seed before testing temporal warm-start versus original initialization on partner retrieval under the same split. |
| Neither arm fits training | Use a fixed tiny temporal gallery to distinguish optimization/capacity problems from generalization. |

Temporal retrieval can exploit slow drift, task state or filtering across adjacent boundaries.
Success is not evidence of interbrain coupling and is not a necessary biological prerequisite for
partner retrieval. The purpose is a manageable intermediate test of transferable learning.
