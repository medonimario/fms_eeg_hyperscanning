# EEG-to-behaviour retrieval: implementation status

Updated 11 November 2026.

| Step | Status |
|---|---|
| 1. Research and design | First draft complete: [research](RESEARCH.md), [plan](PLAN.md) |
| 2. EEG–movement audit and preparation | Implemented and synthetically tested; real-data cluster audit pending |
| 3. Trial splits and fixed retrieval galleries | Next, after reviewing preparation |
| 4. Frozen LaBraM extraction and training-only centering | Planned |
| 5. Small contrastive heads, regression baseline and controls | Planned |
| 6. Whole-trial retrieval and matched mTRF comparison | Planned |
| 7. Partner-specific and unseen-dyad experiments | Later |

## Preparation contract

Preparation retains all three conditions. It excludes the entire paired trial when any of its three six-second
sections is marked bad. Indexing uses the confirmed local contrastive dictionary convention: `trial = index // 3`,
`section = index % 3`. This implements whole-trial rejection but intentionally differs from the historical mTRF
notebook's unusual modulo-20 indexing. `trials.csv` records both decisions; completion metadata counts disagreements.
Do not compare this cache directly with historical mTRF scores without matching trial identities and galleries.

EEG preparation reuses the existing contrastive reader: strict paired events/selections/time axes, canonical 64
channels, float32 microvolts, resampling each four-second window to 200 Hz. No extra filtering, referencing,
channel z-scoring or model scaling is applied. Divide by 100 exactly once when forwarding LaBraM later.

Each retained `[2,20)`-second trial supplies eight windows starting at 2,4,6,8,10,12,14,16 seconds in the original
condition epoch. All windows share a trial ID. `windows.csv` includes EEG array indices and matching movement
slice indices. These are correlated observations, not independent trials. Splits remain explicitly unassigned.

Movement comes directly from your existing trusted `phase_list.pickle`: vertical Hilbert phase, both participants,
240 Hz. Apply the same two-second crop and pair-2450 extra-trial correction as mTRF. Store one `(2,4320)` phase array
per retained paired trial, retaining original precision. Do not interpolate wrapped phase or compute a spectrogram.
Sine/cosine and any later resampling will be explicit downstream representations. Practice removal handles pair
2030's twelve epochs; other pairs have six. Pair 2020 stays in the cache with `main_summary_eligible=false`.

The original 52-dyad cohort is required. Missing artifact keys, ambiguous a/b filenames, inconsistent paired EEG,
wrong movement shapes and nonfinite signals fail explicitly. The trusted phase list has no embedded trial IDs;
its cohort ordering is the existing `mTRF.common.PAIRS` contract. Use the original output from `mTRF.movement`, not
an independently reordered pickle. Preparation verifies structure, not the original CSV trigger reconstruction.

## Run on the cluster

Your movement output can be reused directly; do not rerun movement extraction.

From the repository root, after deploying the new package and wrapper:

```bash
mkdir -p outputs errors
bsub < src/cl_behaviour/hpc/circling_prepare.lsf
```

The default operation is a metadata/phase audit. It reads EEG headers, validates movement and hashes sources, but
never extracts EEG signals. The wrapper defaults to your supplied movement path:
`/work3/s204684/Thesis/circling_eeg/movement/runs/circling_movement_01/phase_list.pickle`.

Review `audit.json`, `census.csv`, and `trials.csv`, particularly exclusion disagreements and retained counts.
Then submit signal preparation:

```bash
STAGE=prepare bsub < src/cl_behaviour/hpc/circling_prepare.lsf
```

Outputs are new directories under `/work3/s204684/Thesis/data/processed/`, named
`cl_behaviour_audit_JOBID` or `cl_behaviour_prepare_JOBID`. Override `OUTPUT_ROOT`, `DATA_DIR`, `PHASE`,
`BAD_SEGMENTS`, `PROJECT_DIR`, or `VENV` in the submission environment if necessary.
HPC scripts are Git-ignored in this repository: copy the wrapper explicitly when deploying.

Equivalent direct invocation:

```bash
PYTHONPATH=src python -m cl_behaviour audit \
  --data-dir /work3/s204684/Thesis/circling_eeg \
  --phase /work3/s204684/Thesis/circling_eeg/movement/runs/circling_movement_01/phase_list.pickle \
  --output /path/to/new-audit
```

Use `prepare` instead of `audit` for the signal cache. Supply `--bad-segments` if needed. Never reuse an output
directory: incomplete runs deliberately have no completion marker. A completed preparation has `cache.json`,
manifest/source/array hashes, `files.csv`, `trials.csv`, `windows.csv`, `census.csv`, and per-dyad EEG/phase arrays.

Bring back the small JSON/CSV audit and preparation outputs for review; the large arrays can remain on HPC.
No new packages are required beyond the existing Braindecode/mTRF environment.

Run targeted checks with `PYTHONPATH=src python -m pytest tests/cl_behaviour`.
