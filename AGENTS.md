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
