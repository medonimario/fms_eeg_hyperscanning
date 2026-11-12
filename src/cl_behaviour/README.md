# EEG-to-behaviour retrieval: implementation status

Updated 12 November 2026.

| Step | Status |
|---|---|
| 1. Research and design | First draft complete: [research](RESEARCH.md), [plan](PLAN.md) |
| 2. EEG–movement audit and preparation | Completed on cluster; preparation 30004124 metadata verified locally |
| 3. Trial splits and fixed retrieval galleries | Split 30017853 verified; combined closed conditions |
| 4. Frozen LaBraM and movement features | Implemented; cluster smoke check next; centering deferred to training |
| 5. Small contrastive heads, regression baseline and controls | Planned |
| 6. Whole-trial retrieval and matched mTRF comparison | Planned |
| 7. Partner-specific and unseen-dyad experiments | Later |

## Preparation contract

Preparation retains all three conditions and now uses **the colleague's mTRF notebook exclusions exactly**:
`((index - 1) % 20) + 1`, compared directly with zero-based `range(20)`. Both participants and movement inherit
that decision. This replaces the incorrect `index // 3` assumption used by preparation format version 1.
Use only new format-version-2 audits/caches with `artifact_policy=mtrf-notebook-exact`.

The notebook's boundary behavior is deliberately preserved: indices 0, 20, 40 and 60 map to 20 and therefore do not
exclude a trial. The supplied dictionary contains none of those indices; `artifact_unmatched_indices` reports
any such entries in future inputs. This is exact reproduction of the authorized reference, not a new indexing
convention. The manifest records contributing `artifact_indices`, without inferring within-trial bad sections.
`historical_exclusion_disagreements` must be zero. The old audit `cl_behaviour_audit_30001599` is superseded.

After deploying updated code, rerun the audit in a fresh job before preparation. With the same dictionary and
cohort, expect 3,055 retained paired trials and 24,440 paired windows (65 excluded trials).

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

Review `audit.json`, `census.csv`, and `trials.csv`, requiring zero exclusion disagreements and checking retained counts.
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

## Freeze splits and galleries (current step)

Preparation `30004124` passed local hash, census, notebook-dictionary exclusion, complete-window,
repetition-order, array-index and EEG–phase timing/link checks: 3,055 retained paired trials, 65 exclusions,
24,440 paired windows. Large signal arrays were not copied locally and have not been independently rehashed here.
The split command validates metadata only. Future feature extraction must verify array hashes, shapes and finiteness
before using signals.

Deploy the package and explicitly copy the Git-ignored `hpc/circling_split.lsf`, then run:

```bash
mkdir -p outputs errors
bsub < src/cl_behaviour/hpc/circling_split.lsf
```

This short CPU job defaults to preparation `30004124` and the historical contrastive cache
`/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487/cache.json`.
Override `CACHE` or `COHORT_MANIFEST` if those paths differ. No EEG, phase arrays or LaBraM weights are loaded.
The historical cohort manifest is required; we do not regenerate its folds or claim its prior exposure is known.
Fold 1 remains external validation and fold 2 external test; only the other folds supply the first galleries.

Equivalent command:

```bash
PYTHONPATH=src python -m cl_behaviour split --cache /path/to/prepared-cache \
  --cohort-manifest /path/to/original-contrastive-cache/cache.json --output /path/to/new-split
```

Within each development dyad, seed 2042 assigns all 20 repetitions to 12 training, four validation and four locked
within-dyad test repetitions **before artifact filtering**. The RNG is keyed by dyad, so input ordering cannot alter
assignments. Source-row ordering must establish the three consecutive conditions of each repetition. Both members,
all conditions and all eight windows inherit that split. Excluded trials retain assignments for auditability.
Pair 2020 remains available for diagnostics, with `main_summary_eligible=false` for later aggregate reporting.

The default first task is **combined closed1 + closed2 / self**. Each participant has up to 40 distinct
18-second trials (20 per closed condition), not 40 independent experimental repetitions. The same 12/4/4
repetition assignment produces up to **24/8/8 trials** for training/validation/test before exclusions.
Closed1 and closed2 from the same repetition always remain together. Candidates may come from either closed
condition but must belong to the same participant and split. Original condition labels and trial identities
are preserved; windows never cross trial boundaries. Each gallery's
ordered `trial_ids` supplies both queries and candidates; positives are the diagonal. The same gallery applies
separately at each matched window offset and later to whole-trial scoring. Artifacts can reduce validation/test
candidate counts below eight (four for separate-condition galleries); actual K, chance top-1 and chance MRR are recorded. Fewer than two candidates is an
explicit error. Test galleries are metadata only and must not be scored during development.

All eight offsets are preserved. The initial training proposal uses non-overlapping crop offsets 0,4,8,12;
eight-window evaluation/aggregation remains available without changing trial assignments. No features, means,
scales, negative-pair permutations or training schedules are fitted at this step.

Split format version 2 records the selected profile and unambiguous repetition/trial counts.

Outputs: `split.json` (completion/provenance/counts), `splits.csv` (including exclusions), `windows.csv` (inherited
splits and source indices), and `galleries.csv` (two participant-specific galleries per development dyad/split for the combined profile).
Bring back `split.json` and `galleries.csv` for review. Frozen LaBraM and movement feature extraction comes next.
This is a new within-dyad development protocol, not the saved mTRF outer-fold benchmark.

### Combined and separate conditions

`--profile combined-closed` is the CLI and wrapper default. The wrapper also accepts `PROFILE=separate-closed`
or `PROFILE=open`. All profiles use identical repetition assignments at the same seed:

| Profile | Eligible trials | Candidate pools per participant/split | Nominal validation K |
|---|---|---|---|
| `combined-closed` | closed1 + closed2 | One combined closed pool | 8 |
| `separate-closed` | closed1 + closed2 | One pool per condition | 4 each |
| `open` | open | One open pool (previous behavior) | 4 |

Combined galleries increase training examples and candidate counts; this does not guarantee easier retrieval.
Closed1 and closed2 may retain recording-order or condition differences. Future scoring should report each query's
original condition, and the separate profile provides a condition-matched comparison on the same assignments.
`galleries.csv` includes `trial_conditions` aligned with `trial_ids`; its `condition=closed` labels a combined
pool, not a relabeling of source data. Actual K and chance remain explicit (nominal combined validation chance
is 1/8, versus 1/4 for separate galleries). Split eligibility selects data only: it does not yet choose whether
future training uses a shared model or participant-specific models.

Reuse preparation `30004124`; no signal preparation is needed. Run the updated split wrapper into a new output
directory. Earlier open split outputs remain valid for reproducing that profile, but are not the new default.

## Paired feature extraction (current step)

The reviewed split `30017853` has 41 development dyads, 952 training and 322 validation paired trials.
That gives 20,384 participant-windows across all eight offsets. The 317 within-dyad test trials and 11 reserved
dyads are not encoded. Pair 2020 remains flagged for exclusion from later main summaries.

Deploy the package and explicitly copy the ignored `hpc/circling_features.lsf`. Start with:

```bash
mkdir -p outputs errors
bsub < src/cl_behaviour/hpc/circling_features.lsf
```

The default `SMOKE=1` processes two deterministic **training-only** trials from the first development dyad,
including both people and all eight offsets. Review `smoke.json`, `diagnostics.csv`, `trials.csv`, and `windows.csv`.
It checks real-array integrity, selected-signal finiteness, token dimensions, pooled reconstruction and repeated
inference. This is an extraction check, not a tiny-set learning or retrieval result. A smoke directory deliberately
has no `features.json`, so it cannot be mistaken for a full training feature bank.

After that check passes, submit the full training/validation extraction:

```bash
SMOKE=0 bsub < src/cl_behaviour/hpc/circling_features.lsf
```

The wrapper uses preparation `30004124`, split `30017853`, and `models/labram-base.pth`. Override `CACHE`,
`SPLIT_RUN`, `CHECKPOINT`, `OUTPUT_ROOT`, or `BATCH_SIZE` as needed. It requests one V100 and four hours (a budget,
not a measured runtime). Batch size 8 means 8 paired windows / 16 individual EEG inputs. Reduce it for GPU memory.
Full tokens alone require approximately 4.17 GB; allow additional space for movement arrays and metadata.

Equivalent direct invocation:

```bash
PYTHONPATH=src python -m cl_behaviour features --cache /path/to/preparation \
  --run /path/to/split --checkpoint models/labram-base.pth --device cuda \
  --batch-size 8 --smoke --output /path/to/new-smoke
```

Remove `--smoke` for full extraction. Never reuse an output directory; failure leaves no completion marker.

### Feature contracts

- Frozen original pretrained LaBraM, evaluation/inference mode, float32, microvolts divided by 100 exactly once.
  No additional EEG filtering or scaling. Each output token array has shape `(paired_windows, 2, 64, 4, 200)`.
  Tokens are channel-major/time-minor, before `fc_norm`, with CLS excluded. Reconstructing original pooled
  features as `fc_norm(mean(tokens))` is checked on every forward call. `pooling_norm.npz` saves its weight,
  bias and epsilon so pooled/temporal/spatial/full readouts can be derived without rerunning LaBraM.
- Self movement: sine then cosine of the original phase, followed by `scipy.signal.resample_poly` with ratio
  5/24, Kaiser beta 5 and line padding. Filtering is applied to each complete 18-second trial independently,
  never across trials or splits. Outputs are float32 `(paired_trials, 2 members, 2 components, 900 samples)`
  at 50 Hz. Wrapped angles are never resampled. Filtered components are not renormalized onto the unit circle.
- `windows.csv` maps each paired EEG feature row to a movement trial and a half-open 200-sample slice at the
  same offset. All eight windows remain available; original condition, repetition, absolute event metadata,
  participant-axis convention, eligibility and split identities are preserved. Overlaps are not independent trials.
- No centering, standardization, projections, losses, soft targets or trainable behaviour encoder are fitted.
  Raw features are reusable. Participant-specific training means remain the proposed initial centering policy,
  to be fitted for the selected readout and training offsets during the training step.

Preparation and split manifests are hashed and the split/gallery/window contracts independently reconstructed.
Only permitted dyad files are opened. Whole-file hash verification necessarily reads bytes belonging to unused
trials within those dyads; only selected training/validation slices are materialized for feature computation and
checked finite. No test trial is forwarded; no reserved-dyad signal file is opened.

Full outputs include `features.json`, `diagnostics.csv`, `trials.csv`, `windows.csv`, `pooling_norm.npz` and
per-dyad token/movement arrays. Completion records input/output hashes, checkpoint identity, normalization,
resampling, dimensions, software versions, counts and reconstruction error. Bring back the JSON and diagnostic
CSV first; arrays can stay on HPC. Successful extraction enables the next **synthetic/tiny-fit and CL training**
step; it does not establish EEG-to-movement predictability.

### Later soft-target training experiment

The user suggested [Truong et al., arXiv:2610.09287, Appendix B](https://arxiv.org/html/2610.09287v1).
Their soft-target CLIP uses similarities between centered, frozen video-teacher embeddings; temporal closeness
is not itself the weighting formula. A temporal buffer is a separate mechanism. For movement, later compare
hard correspondence with fixed movement-similarity soft targets and, separately, time-distance weighting within
an explicitly defined continuous trial. Do not treat equal offsets in different trials as temporal neighbors,
or infer similarity from closed1/closed2 labels alone. Preserve held-out galleries and keep target construction
independent of evolving learned heads. The current cache preserves the information needed without choosing a loss.
