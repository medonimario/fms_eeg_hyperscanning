# Project 2: paired Circling data

Stage 0 provides an audit and paired cache for the **local dataset used by Project 1**. Training is not implemented.
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
selections must be contiguous from zero, and event samples strictly increase. This conservative first version rejects
previously dropped/reindexed epochs, even if both members dropped the same ones, because the dictionary's trial
numbering could otherwise shift. Resolve those cases from original acquisition provenance before adding an adapter;
never just pair the nth surviving epoch. Matching arrays alone cannot detect an upstream reset of both selections,
so the local source provenance still needs review.

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
