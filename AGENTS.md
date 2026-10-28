> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* The project uses `pip` for management of packages. This means:
  * To install packages, use `pip install <package-name>`.
  * To run Python scripts, use `python <script-name>.py`.
  * To run other commands related to Python, use `python <command>`.
* The project uses `pytest` for testing. To run tests, use `pytest tests/`.
* The project uses `ruff` for linting and formatting:
    * To format code, use `ruff format .`.
    * To lint code, use `ruff check . --fix`.
* The project uses `invoke` for task management. To see available tasks, refer to the `tasks.py` file.
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints
* Do not add inline comments unless absolutely necessary.

# Documentation

* If the project has a `docs/` folder, update documentation there as needed.
* In this case the project will be using `mkdocs` for documentation. To build the docs locally, use
    `uv run mkdocs serve`
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.

# Braindecode classification commands

* Set `PYTHONPATH=src` and use `python -m labram_classification --help` for production preprocessing and training.
* See `src/labram_classification/README.md` for the dataset contracts and HPC workflow.
* Run targeted regression tests with `python -m pytest tests/labram_classification tests/labram_classification_lessons`.
* The former tutorial package is now `src/labram_classification_lessons`; production must not import it.
* Optional W&B: `wandb login` authenticates online logging; `wandb sync <offline-run-directory>` uploads an offline run. Set `WANDB_MODE` to online, offline, or disabled.
* Classification `requirements.txt` targets Linux/Python 3.11/CUDA 11.8. Keep application pins grounded in the local snapshot and Torch/TorchAudio matched. `requirements-local-freeze.txt` is the unmodified local pip freeze, with provenance in `requirements-environment.json`; keep resolved HPC snapshots separate.
* Generate ten-fold mean ± SD figures with `python -m labram_classification plot --output <experiment-directory>`. Test curves are diagnostics only; validation balanced accuracy remains the sole checkpoint-selection metric.
* Run the CPU band-power comparator with `python -m labram_classification baseline --cache <cache> --output <new-experiment>`. See `src/labram_classification/BASELINES.md`; fit feature standardization only on training groups and select regularization only on validation groups.
* Train EEGNet from scratch with `python -m labram_classification cv --model eegnet --cache <cache> --output <new-experiment>`. See `src/labram_classification/EEGNET.md` and `hpc/eegnet_cv.lsf` under that package. Preserve separate model profiles: existing commands default to pretrained LaBraM; EEGNet never reads a checkpoint.
* Train random LaBraM with `python -m labram_classification cv --model labram --initialization random --cache <cache> --output <new-experiment>`. See `src/labram_classification/RANDOM_LABRAM.md`. Random runs must never read or hash pretrained weights; preserve identical architecture and record initialization in each run.

# Hyperscanning contrastive data commands

* Set `PYTHONPATH=src`; use `python -m hyperscanning_contrastive audit --data-dir <local-circling> --output <new-audit>` for Stage 0 metadata checks, and `python -m hyperscanning_contrastive prepare --data-dir <local-circling> --output <new-cache>` to save paired signals. Supply `--bad-segments <trusted-local-pickle>` when it is not in the dataset parent directory.
* This uses the local Project 1 Circling dataset and its confirmed **six-second** artifact dictionary, not the public release. Exclude artifact-overlapping intervals symmetrically across each dyad. See `src/hyperscanning_contrastive/README.md` for maximal clean non-overlapping four-second windows and strict pairing checks.
* Run targeted tests with `python -m pytest tests/hyperscanning_contrastive`. The `overfit` command is a development-only training diagnostic; Stage 2 SSL and Stage 3 projection training are available; full CV remains unimplemented.
* The local Circling data are on the HPC at `/work3/s204684/Thesis/circling_eeg/`. Submit the CPU audit with `bsub < src/hyperscanning_contrastive/hpc/circling_data.lsf`; use `STAGE=prepare bsub < src/hyperscanning_contrastive/hpc/circling_data.lsf` after reviewing the audit. Create `outputs/` and `errors/` before submission. Each job writes a new output directory keyed by its job ID.

* Run Stage 1 synthetic CPU controls with `PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive controls --output <new-directory>`. This checks loss/retrieval and synthetic fitting, not LaBraM or EEG performance.

* Run `PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive overfit --cache <paired-cache> --output <new-run>` for a frozen-LaBraM tiny-set diagnostic. Use `--initialization random` without checkpoint access; `--train-backbone --device cuda` enables encoder updates. See the package README and `hpc/circling_overfit.lsf`.

* Stage 2: `PYTHONPATH=src python -m hyperscanning_contrastive ssl --cache <cache> --output <new-run> --device cuda` trains single-person augmented-view SSL. Defaults hold out fold 1 for validation and fold 2 for test; fold 0 remains training. `--initialization random` never reads pretrained weights. Use `ssl-evaluate --cache <cache> --run <completed-run> --output <new-test-run> --device cuda` only after locking settings. See README for fixed galleries and validation-only MRR selection; GPU wrapper: `hpc/circling_ssl.lsf`.

* Validate Stage 2 correspondence with `PYTHONPATH=src python -m hyperscanning_contrastive ssl-correspondence --cache <cache> --run <completed-run> --output <new-directory> --device cpu`. This uses validation only and uniform permutations with fixed points allowed; it is an engineering control, not a hyperscanning significance test. HPC wrapper: `hpc/circling_correspondence.lsf`.

* Stage 3 frozen baseline: `PYTHONPATH=src python -m hyperscanning_contrastive partner-baseline --cache <cache> --output <new-run> --initialization pretrained --device cpu`; repeat with random. Validation fold 1 only, aligned opposite participants, matched condition/offset trial galleries, no adaptation or test scoring. See README and `hpc/circling_partner_baseline.lsf`.

* Train Stage 3 frozen-feature projections with `PYTHONPATH=src python -m hyperscanning_contrastive partner-train --cache <cache> --output <new-run> --device cuda`. Each initialization runs true and fixed shuffled-training-pair arms from identical heads/sampling schedules; only projections train. Validation fold 1 selects checkpoints; test fold 2 is not scored. See `hpc/circling_partner_train.lsf` and package README.

* Stage 3 `partner-train` now logs fixed train/validation retrieval, cosine margins, normalized embedding spread/effective rank, and gradient/update norms by default. Collect these with the existing wrapper in a new run; inspect both `true/history.csv` and `shuffled/history.csv` before changing optimization settings.

* For the Stage 3 geometry ablation, pass `--projection-profile centered-linear` or set `PROJECTION_PROFILE=centered-linear` for the partner-training wrapper. Fit the global feature mean on training windows only; save and reuse it for all splits. The default `legacy-mlp` remains unchanged. Compare raw pooled, centered pooled, initial head, and true/shuffled selected heads.

* Limited Stage 3 adaptation: `PYTHONPATH=src python -m hyperscanning_contrastive partner-adapt --cache <cache> --output <new-run> --device cuda` trains the final two LaBraM blocks, pooling normalization, and original MLP. Defaults use encoder LR `1e-5`, head LR `1e-3`, matched true/shuffled arms, validation fold 1, and untouched test fold 2. See README and `hpc/circling_partner_adapt.lsf`; copy ignored HPC scripts explicitly when deploying via Git.

* Plot recorded partner results without EEG/model loading: `PYTHONPATH=src python -m hyperscanning_contrastive.plotting --run <completed-run> --output <figure-directory>`. Requires Matplotlib; exports retrieval/geometry PNG and PDF curves for true and shuffled arms.

* Within-dyad Stage 3: add `--validation-protocol within-dyad --validation-trials 8 --split-seed 2042` to `partner-adapt`. Hold out paired whole trials within original training dyads; never forward fold 1 external-validation or fold 2 test EEG. Use separate train/validation feature banks despite shared dyad IDs. See `hpc/circling_partner_within_dyad.lsf`, which runs both arms and generates plots. Preserve the unseen-dyad default and trial-split manifest.

* Spectral partner comparator: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive partner-spectral --cache <cache> --run <completed-within-dyad-run> --output <new-run>`. CPU-only; verifies reference split/gallery/mapping hashes, fits standardization only on training features, compares direct log-band-power retrieval with matched true/shuffled shared linear projections. No checkpoint reads or external/test feature extraction. See `hpc/circling_partner_spectral.lsf`; plotting also accepts `spectral.json`.

* Stage 2.5 temporal bridge: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive temporal-bridge --cache <cache> --output <new-run> --device cuda`. Same-person non-overlapping 2–6/6–10-second windows; fresh whole-trial within-dyad split after clean-window intersection, reserved folds 1/2 untouched. Frozen features, training-only standardized linear head, full training trial galleries, four stratum losses/update, matched true/shuffled arms at temperatures .05/.1/.2. Generates plots automatically. See README and `hpc/circling_temporal_bridge.lsf`; report actual negatives separately from update batch size.

* Temporal optimization sweep: add `--temperatures 0.05 --learning-rates 0.0001 0.0003 0.001 --weight-decays 0.01 0.1` to `temporal-bridge`, or submit `hpc/circling_temporal_tuning.lsf`. Extracts features once, keeps matched arms and manifests, writes `comparison.csv`, `selected_experiment`, and `selected_settings`; defaults preserve original optimization.

* Unseen-dyad temporal evaluation: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive temporal-transfer --cache <cache> --run <completed-temporal-run> --output <new-run> --experiment temperature_0.05 --device cuda`. Defaults lock the original temperature-.05 run configuration; only fold 1 EEG is forwarded, fold 2 remains untouched. Verifies provenance and restores training-fitted statistics/selected heads; no retraining or external checkpoint selection. See README and `hpc/circling_temporal_transfer.lsf`; automatic paired-dyad plots and differences accompany `transfer.json`.

* All-window partner experiment: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive partner-windows --cache <cache> --output <new-run> --device cuda --validation-trials 8 --epochs 20 --groups-per-update 4`. Splits verified whole trials before grouping every clean window by dyad/condition/exact interval; opposite-participant positives and other-trial negatives. Fresh frozen-LaBraM standardized linear head; matched true/shuffled arms at temperature .05/LR .001/decay .01. Reports singleton exclusions, variable-gallery chance, interval/condition/dyad scores and plots. See README and `hpc/circling_partner_windows.lsf`. Legacy cache readers keep offset-zero behavior; the new reader opts into all windows.


### Open-only partner-window experiment

`partner-windows` now defaults to `--conditions open`. Only open windows enter feature standardization,
true/shuffled training, baseline scoring, and validation checkpoint selection. Whole-trial splits are still
assigned across all cache conditions before filtering, preserving the earlier open trial assignments and
candidate galleries with the same split seed. Manifests retain excluded conditions with
`reason=condition_not_selected`; report coverage counts only selected conditions. No cache rebuild is needed.
Original external-validation/test dyads remain reserved. Mean-pooled frozen LaBraM features are unchanged.

Submit the dedicated wrapper from the repository root:

```bash
CACHE=/work3/s204684/Thesis/data/processed/circling_contrastive_prepare_29758487 \
  bsub < src/hyperscanning_contrastive/hpc/circling_partner_windows_open.lsf
```

Create `outputs/` and `errors/` first. HPC scripts are Git-ignored: copy this wrapper explicitly when deploying.
The existing `circling_partner_windows.lsf` explicitly retains all three conditions for reproduction;
CLI reproduction uses `--conditions closed1 open closed2`.
The open-only run retains 20 epochs, four independent interval sets per update, temperature 0.05, and LR 0.001.
It visits each usable open training set once per epoch, so total updates decrease with the number of conditions.
Standardization and shuffled mappings are refitted for this open-only run. Compare against the prior run's
**open** metrics, not its three-condition macro average. Automatic PNG/PDF curves are included.

* Patch readouts: `PYTHONPATH=src python -m hyperscanning_contrastive partner-patches --cache <cache> --output <new-run> --device cuda` runs open-only pooled/spatial/temporal/full frozen readouts with matched true/shuffled arms. See package README and `hpc/circling_partner_patches.lsf`. Preserve pre-fc_norm tokens, check pooled reconstruction, and retain identical split/gallery/mapping manifests. Outputs include `patches.json`, `comparison.csv`, and PNG/PDF comparisons.

* Partner readout optimization: `partner-windows` and `partner-patches` accept singular `--temperature`, `--projection-lr`, and `--weight-decay` (defaults 0.05/0.001/0.01). `partner-patches --readout-profiles full` selects a subset; default is all four. HPC wrappers expose TEMPERATURE/PROJECTION_LR/WEIGHT_DECAY and patch READOUT_PROFILES. Plural tuning flags remain specific to temporal sweeps.

* Video Stage 0: `PYTHONPATH=src python -m hyperscanning_contrastive video-audit --data-dir <video-root> --output <new-audit>` scans all trusted NPZ files; `video-prepare` also writes continuous 200 Hz microvolt arrays. Separate from Circling. See package README and `hpc/video_data.lsf`; VIDEO_DATA_DIR is required on HPC. Do not infer babble identity from A/B, shift timelines after rejection, or treat unassigned Stage0 windows as a training split. Target tests: `python -m pytest tests/hyperscanning_contrastive/test_video_data.py`.

* Video Stage 1: `PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m hyperscanning_contrastive video-baseline --cache <video-cache> --output <new-run>` runs solo/video0 held-time waveform/CorrCA controls on CPU. See README and `hpc/video_baseline.lsf`; reserved participant signals must not be accessed. Fixed K=4 galleries, 4s boundary guard, training-only covariance fit, no significance test.

* Video frozen comparison: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive video-labram --cache <video-cache> --run <completed-video-baseline> --output <new-run> --device cpu`. Reuse exact saved galleries/splits; no training or reserved-person encoding. Original pooled features, cosine retrieval, automatic comparison plots. See `hpc/video_labram.lsf`; inference batch size only controls memory. Random initialization never reads/hashes weights.

* Video CNN: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive video-cnn --cache <video-cache> --run <completed-video-baseline> --output <new-run> --device cpu`. Small spatial-temporal CNN, training-only channel scaling, disposable tiny-fit gate, matched true/batch-misaligned arms and original K=4 galleries. See README and `hpc/video_cnn.lsf`; `--cnn-temperature`/`--cnn-lr`/`--cnn-weight-decay` control optimization. Never encode reserved people.

* Video LaBraM heads: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m hyperscanning_contrastive video-projection --cache <video-cache> --run <completed-video-baseline> --output <new-run>`. Frozen original pooled/temporal readouts, raw/global/participant centering diagnostics and matched contrastive heads. Participant means use matched training moments only and imply calibration. See README and `hpc/video_projection.lsf`; flags `--video-head-lr`, `--video-temperature`, `--video-weight-decay`, `--centerings`. No CNN signal normalization or reserved-person forwarding.

* Video spatial/full readouts: add `--video-readouts spatial full` to `video-projection`, or submit `hpc/video_spatial_projection.lsf`. Existing defaults remain pooled/temporal. `--diagnostics-only` (wrapper DIAGNOSTICS_ONLY=1) skips all head training. Spatial/full heads use compact shared-token projections and signed channel mixtures, preserve channel-major/time-minor ordering, and reuse original split/gallery/centering contracts.

* Independent video runs: `video-individual --cache <cache> --run <original-video0-baseline> --output <new-run> --video-id <3..10> --video-noise combined --video-readouts spatial full` through `python -m hyperscanning_contrastive`. `hpc/video_individual.lsf` is an eight-job CPU array. Keep inherited participant reservations, movie-specific held-time splits and fresh heads/statistics. Combined low/high solo viewers are intentional; no cross-movie batches or training. Each movie writes separate baseline/ and projection/ directories.

* Identity weighting: `video-projection --video-head-type feature-weights --video-readouts full --centerings global` reuses an existing movie baseline. Frozen 256x200 tokens, 200 shared positive identity-initialized weights, no compression; epoch-zero scores must match direct retrieval. Video 3 wrapper: `hpc/video_feature_weights.lsf`; outputs include per-arm best/last weights and learning curves. See package README.
