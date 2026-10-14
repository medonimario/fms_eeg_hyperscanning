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
