# Circling movement and forward-mTRF replication

Run from the repository root with Python 3.11+ and `PYTHONPATH=src`. The two modules are counterparts to the two example
notebooks. The active fork lives in `src/mTRF/mtrf_modified/`, beside the entry points. Its model, matrices and active
statistics module were copied from the supplied fork (version label 2.1.2) with only the internal imports made relative.
The original license is included. Both serial and Joblib workers import this copy. The archived statistics variants are
unneeded. The saved notebook does not identify the exact revision that generated its original CV pickle.

## Inputs

Set `CIRCLING_DATASET_PATH` in the repository-root `.env` to the parent of these files. An exported environment variable
takes precedence over `.env`, and `--data-dir` overrides either. Output directories are specified separately.

```text
circling_eeg/
  behaviour_signal/pair2010/polhemus_001_20240424_150636.csv
  preprocessed/pair2010a_ave_epo.fif
  preprocessed/pair2010b_ave_epo.fif
  dict_with_short_bad_epochs.pickle
```

The EEG reader also accepts an underscore or hyphen between pair number and member letter. Each member must resolve to
exactly one file. Use the original trusted six-second bad-segment dictionary for this local Circling data. If it is
elsewhere, pass `--bad-segments` or export `BAD_SEGMENTS` for the batch job. Only load trusted local pickles.

## Run the two stages

```bash
pip install -r src/mTRF/requirements.txt
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m mTRF.movement --output runs/circling_movement --report
python -m mTRF.eeg fit --phase runs/circling_movement/phase_list.pickle --output runs/circling_mtrf --jobs 4
```

Movement processes the 52 dyads in notebook order and writes `phase_list.pickle`, `plv_results.pickle` and
`ccor_results.pickle`. The optional report adds condition summaries, paired contrasts, plots, the truncated PLV analysis
and the seed-123, 1,000-permutation Vision test. The phase list keeps pair 2450's 21st trial. You can also pass your
colleague's original **vertical** phase_list.pickle directly to the EEG fit. The horizontal file is a different signal.

For one dyad first, append `--pair 2010` to the fit command. Each completed dyad writes an atomic pair pickle.
Use a fresh EEG output directory per experiment. To retry an interrupted dyad, run the identical command with that
pair and the same inputs. The full-cohort command also collects results. For separate dyad jobs run collect after all
52 finish. Collection fails if any shard is missing.

```bash
python -m mTRF.eeg collect --output runs/circling_mtrf
```

## LSF

Set the dataset root in the HPC checkout's `.env`. Export the job settings below in the submission shell,
using absolute paths on the cluster. Install the requirements in `VENV` before submission.

```bash
export PROJECT_DIR="$PWD"
export VENV=/path/to/your/python-environment
export MOVEMENT_OUTPUT=/path/to/new/movement-run
export EEG_OUTPUT=/path/to/new/eeg-run
mkdir -p outputs errors
bsub < src/mTRF/hpc/movement.lsf
```

After the movement job succeeds, submit `bsub < src/mTRF/hpc/eeg.lsf`. The CPU array assigns one pair per task,
four slots per pair and up to eight pairs concurrently. Each task uses the same movement pickle and writes a separate
shard. After all tasks succeed, activate the same environment, then run
`PYTHONPATH=src MPLBACKEND=Agg python -m mTRF.eeg collect --output "$EEG_OUTPUT"` from the repository root.

## Replication details

- Movement uses the notebook's CSV column correction, interpolation, vertical X position, within-trial z-score,
  10 Hz FIR and Hilbert phase on 20-second, 240 Hz segments. The CSV for pair 2020 has missing trial triggers.
  The supplied onset table contains the 60 experimental row offsets (after interpolation). These were recovered by
  aligning the CSV against the colleague's saved vertical phase list, and verified sample-for-sample for every
  condition, trial and participant. The CLI checks the CSV SHA-256 before applying them. This avoids needing raw BDF
  on HPC. Local verification regenerated identical phase arrays for all 52 dyads, including 2020.
- EEG uses the full retained trials. Crop the first two seconds and drop six practice epochs (12 for pair 2030).
  For pair 2450, delete zero-based movement trial 10 in each condition before alignment. Apply the same selected trials
  to both movement tracks and both EEG recordings, resample EEG to 240 Hz, and z-score each channel over time within
  each trial. Each aligned trial has 4,320 samples. No LaBraM preprocessing, new EEG windowing or contrastive cache.
- **Bad-segment indexing follows the EEG notebook literally.** It maps a segment index `i` to
  `((i - 1) % 20) + 1` and compares that value directly to zero-based `range(20)`. Thus returned value 20 removes
  nothing and trial index 0 is never removed. This differs from the other project's segment-to-trial convention.
  Changing the mapping changes the replication.
- Fit Full, SelfOther, SelfCoordination and OtherCoordination separately for each person and condition. Sine/cosine
  encode self and partner phase, with cosine of wrapped relative phase as coordination. Preserve the fork's whole-trial
  shuffle, 5 outer / 4 inner folds, seed 123, lags −250 to +300 ms, 10 penalties from 0.1 to 1e6, correlation-based
  penalty selection and residual-based outer R². Parallelism affects runtime. Keep BLAS threads at one per worker.
- Collection produces the two original-format result files, `trf_long_all_rsq_sin_cos_2.pickle` (fold correlation,
  penalty and R²) and `trf_visualization_sin_cos_2.pickle` (Vision full-data refit per electrode at the median
  fold penalty). CSVs and headless figures accompany them. Main participant contrasts omit both members of pair 2020,
  apply the notebook's Fisher transform to fold R², then subtract reduced from full. Paired p-values are reported
  with the notebook's literal multiplication by three. For comparison, the saved notebook's median partner ΔR² in
  Vision is 0.000090. The separate topography cell literally drops participant positions 1 and 2 instead of 2 and 3.
  The scalp figure preserves that slice, and the map pickle retains everyone. Headless plots use simplified styling.
- This covers the movement signals, movement synchronization and EEG encoding results. The original notebooks retain
  the exploratory displays and subsequent social-network/personality analyses.

Run `python -m pytest tests/mtrf` for regression checks on artifact indexing, pair-specific trial handling, the
actual nested fitting entry point and group aggregation using synthetic FIFs.

## Held-out movement retrieval baseline

The retrieval command asks which candidate movement recording belongs to a query EEG trial. It uses the saved **Full**
model configuration (self sine/cosine, partner sine/cosine and coordination). It retrieves the entire aligned movement
record, so success can depend on self movement as well as partner movement. This is a first benchmark for comparison
with a contrastive model using the same inputs, splits and galleries.

The original result files save penalties and fold scores, but omit fitted weights and trial predictions. Retrieval
therefore needs the original preprocessed FIFs, phase pickle and bad-segment dictionary. For each original outer fold,
it reconstructs the seed-123 trial split and refits on training trials with the **saved penalty**. It skips the inner
penalty search. Before ranking candidates, it requires the refit to reproduce that fold's saved correlation and R²
(rtol 1e-5, atol 1e-8). A mismatch fails the pair with a diagnostic message.

Each query's candidates are exactly the other held-out trials plus the correct trial from the same participant,
condition and outer fold. All candidates are whole 18-second trials. Each candidate's movement is passed through that
participant's fitted forward mTRF. The score is residual R² against the query EEG, averaged across channels.
Higher scores rank first. No retrieval setting is selected from these test-fold scores.

Top-1 chance is 1/K and MRR chance is (1 + 1/2 + ... + 1/K)/K. With this cohort's retained-trial counts, K is 3 or 4,
so top-1 chance is 33.3% or 25% per query. Exact ties receive the expected score for uniformly random tie-breaking.
Chance is averaged with the same weights as the measured scores. Collection first averages queries per person, then
the two people per dyad, then dyads per condition. Pair 2020 remains in detailed outputs and is excluded from group
summaries, matching the main replication. This is a descriptive comparison with chance, without a significance claim.

For a single-pair check, activate the same environment as the original fit and run from the repository root:

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m mTRF.retrieval run \
    --phase /path/to/original/movement-run/phase_list.pickle \
    --run /path/to/completed/circling_mtrf_01 \
    --output /path/to/new/circling_mtrf_retrieval_01 \
    --pair 2010 --jobs 4
```

For all pairs, export MOVEMENT_OUTPUT and EEG_OUTPUT as in the original jobs, plus RETRIEVAL_OUTPUT pointing to a
separate new directory. The HPC directory is Git-ignored; copy `src/mTRF/hpc/retrieval.lsf` explicitly when deploying.
Submit `bsub < src/mTRF/hpc/retrieval.lsf`. This uses the requested XeonGold6226R CPU array,
four workers per dyad, a four-hour limit, and at most eight concurrent dyads. Each pair requires 30 refits
(two people × three conditions × five folds), with no new cross-validation search. Retry a failed task using the same
inputs and its `--pair` value, writing to the same retrieval directory.

After all 52 tasks finish:

```bash
PYTHONPATH=src python -m mTRF.retrieval collect --output "$RETRIEVAL_OUTPUT"
```

Collection requires only the saved retrieval shards and can run locally after copying them back. Outputs include
`summary.csv`, `retrieval.png`, `participants.csv`, `dyads.csv`, `queries.csv` and `candidate_scores.csv`.
Each pair pickle preserves train/test trial IDs, all query-by-candidate score matrices, original/refitted fold metrics,
selected penalties and EEG source hashes. `protocol.json` records input/code hashes and the scoring convention.
Collection rejects missing or incompatible shards. Original fitted weights are still not retained.

The retrieval gallery and query lists are suitable for reusing unchanged in a future contrastive benchmark.
Developing that model on these displayed results will require a separate final evaluation for an unbiased
model-comparison claim.
