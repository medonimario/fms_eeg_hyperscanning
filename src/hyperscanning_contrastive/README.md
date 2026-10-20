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

Stage 3 partner alignment and trial-level surrogate evaluation are not implemented by these commands.

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

The next Stage 3 step is a learned shared projection and then limited encoder adaptation, compared against these
frozen baselines. Partner training is not implemented by `partner-baseline`.
