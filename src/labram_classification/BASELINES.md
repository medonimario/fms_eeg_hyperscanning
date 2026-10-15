# Band-power baselines for the LaBraM comparison

The first baseline is **log-band-power features → StandardScaler → L2-regularized logistic regression**.
It runs on CPU, uses the exact prepared EEG windows and group partitions used by LaBraM, and produces
comparable fold metrics and per-window predictions. It works with MG, FG feedback, and Circling caches.
No original recordings, GPU, or pretrained model checkpoint are required.

## Why these features?

A whole-window power spectrum is a compact starting point: it tests whether channel-specific spectral
power alone separates the conditions. A time-frequency representation can additionally preserve
changes within each four-second window. Both variants are available as separate, explicitly named
experiments; neither treats a spectrogram as an image for another neural network.

| Mode | Representation | Features per EEG window |
| --- | --- | --- |
| `bandpower` (default) | Welch spectrum over the four-second window, integrated over five bands | 64 × 5 = 320 |
| `time-bandpower` | Band power retained separately for intervals 0–2, 1–3, and 2–4 seconds | 3 × 64 × 5 = 960 |

The time intervals are offsets from each cached window's beginning, not absolute event timestamps.
The time-resolved variant is a coarse, band-integrated short-time Fourier power representation. It uses
the same spectral estimator as the simple baseline, isolating the effect of retaining temporal bins.

Five fixed bands are used: delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, and low gamma 30–45 Hz.
These are conventional feature labels, not evidence that each band reflects a specific brain process.
Interpret bands in light of the source preprocessing filters and possible muscle/movement contamination.
No additional filtering, rereferencing, baseline correction, channel selection, or resampling is added.

## Exact extraction procedure

- Read the existing arrays in **microvolts**, without LaBraM's division by 100.
- Use a two-second (400-sample) periodic Hann window and one-second stride, at 200 Hz.
- Remove each spectral segment's mean (`detrend='constant'`). Use a one-sided PSD with density scaling,
  400-point FFT (0.5-Hz frequency spacing), and no extra zero-padding.
- Whole-window mode averages the three modified periodograms, as in Welch's method. Time-resolved mode
  retains them separately.
- Integrate the PSD over each band with the trapezoidal rule, including the exact band-edge bins.
  Shared boundary values are endpoints of adjacent integrals; no full boundary bin is counted twice.
- Apply `log10(max(power, 1e-12))`, where power is in microvolts squared. These are absolute powers,
  not relative powers or decibels.
- Flatten in time-bin, channel, band order. `feature_names.csv` records this ordering.

All of this is a deterministic transform applied independently to each EEG window; computing it once
for the full cache does not fit anything to held-out data. Feature extraction runs in bounded batches,
writing a memory-mapped `features.npy`. Model fitting holds partition feature matrices in RAM, not all EEG.

References: [SciPy Welch PSD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html),
[scikit-learn preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html).

## Splits, fitting, and model selection

For run k, the same cache assigns test fold k, validation fold (k+1)%10, and the other eight folds to
training. The shared fold loader checks group/participant disjointness and the presence of both classes.
Overlapping windows stay together with their entire dyad/triad.

For each candidate C, a fresh scikit-learn pipeline fits StandardScaler and logistic regression only on
training windows. The candidate with the highest **validation balanced accuracy** is retained.
Default C values are 0.01, 0.1, 1, 10, 100; smaller C means stronger regularization, and ties choose the
smallest C. The default solver is LBFGS with L2 regularization and a 2000-iteration limit. All windows
have equal weight; no class weighting or resampling is added. Failure to converge stops the run rather
than silently accepting a poorly fitted candidate; increase `--max-iter` if necessary.

Validation is not merged into training after selection, matching LaBraM's partition usage. Test data
are evaluated only after C has been selected. Predictions use a fixed probability threshold >0.5.
The test set is not used to choose feature mode, bands, C, normalization, or threshold.

The baseline tunes C across five candidates; LaBraM uses its declared training configuration and
validation epoch selection. Report these different selection budgets in a comparison. If you choose
between the two feature modes, use validation results, not whichever test result looks better.

## Run locally or on the cluster

Use the existing project environment (SciPy, scikit-learn, joblib and threadpoolctl are already part of
its dependency set; they are now explicit pins in the production requirements).

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# All ten folds, whole-window spectral features:
python -m labram_classification baseline \
  --cache /work3/s204684/Thesis/data/processed/mg_solo_coordination \
  --output models/results/mg_bandpower_run1

# A separate time-frequency baseline:
python -m labram_classification baseline \
  --cache /work3/s204684/Thesis/data/processed/mg_solo_coordination \
  --output models/results/mg_time_bandpower_run1 \
  --feature-mode time-bandpower
```

The command defaults to all ten folds and automatically writes their summary. For an initial check,
add `--fold 0` and use a separate new output directory. This is a complete single-fold fit, not a
subsampled smoke run. Existing output directories are never overwritten. Interrupted runs currently
require a new output directory; there is no optimizer/feature-cache resume command.

Optional settings: `--c-values 0.01 0.1 1 10 100`, `--max-iter 2000`, `--num-threads 4`,
`--feature-batch-size 64`, `--seed 0`. These options are recorded in each fold configuration.

For LSF, use `hpc/baseline_cv.lsf`, which requests CPU resources and no GPU. Set `PROJECT_DIR`, `VENV`,
`CACHE_DIR`, `OUTPUT_DIR`, and optionally `FEATURE_MODE=time-bandpower` before submission. Create
`outputs/` and `errors/` first. The template leaves the queue unspecified; select the appropriate CPU
queue for your cluster and adjust its illustrative wall-time/memory request as needed.

```bash
mkdir -p outputs errors
export OUTPUT_DIR="$PROJECT_DIR/models/results/mg_bandpower_run1"
export WANDB_MODE=online
export WANDB_RUN_GROUP=mg_bandpower_run1
bsub < src/labram_classification/hpc/baseline_cv.lsf
```

W&B supports the same mode/project/entity/group settings and flags as neural training. Each fold has
its own run. `selection/*` logs training/validation loss and balanced accuracy for each C candidate;
these are model-selection trials, **not epochs**. Final train/validation/test scores and selected C are
logged after selection. Use a distinct W&B group for each feature mode/experiment.

## Saved outputs and comparison

Experiment root:

- `features.npy`: deterministic feature matrix, shared by all folds.
- `features.json`: spectral parameters, cache metadata hash, feature matrix hash and feature count.
- `feature_windows.csv`, `feature_names.csv`: exact row and column identities.
- `fold_metrics.csv`, `summary.json`: final test metrics across all ten folds, compatible with the
  neural pipeline's metric names and sample-SD convention.

Each fold:

- `config.json`: dataset, feature specification, options, versions and fold identity.
- `train_windows.csv`, `valid_windows.csv`, `test_windows.csv`: the same partitions as LaBraM.
- `selection.csv`: training/validation candidate scores.
- `model.joblib`: the selected fitted scaler + logistic-regression pipeline.
- `coefficients.csv`: coefficients in standardized feature coordinates. Positive values favor class 1;
  correlated EEG features mean these weights are not causal explanations.
- `train_predictions.csv`, `valid_predictions.csv`, `test_predictions.csv`: labels/probabilities and provenance.
- `result.json`: completion marker, selected C, metrics and configuration hash.

ROC AUC, PR AUC, average precision, loss, accuracy, balanced accuracy, F1 and confusion matrices use
the same definitions as LaBraM. Match the cache metadata hash and per-fold group/window manifests
before comparing experiments. Baseline performance is measured per window, with the same dependence
between overlapping windows; do not treat windows as independent participants for significance tests.

There are no neural training epochs, so the `plot` learning-curve command does not apply. Use the
baseline's `fold_metrics.csv` alongside LaBraM's for the eventual model-comparison figure.

To predict from a saved fold on new prepared microvolt windows:

```python
import joblib
from labram_classification.baseline import bandpower_features

# Only load your own trusted model files.
model = joblib.load("models/results/mg_bandpower_run1/fold_0/model.joblib")
features = bandpower_features(windows_uv, mode="bandpower")  # shape (N, 64, 800), 200 Hz
probability_class_1 = model.predict_proba(features)[:, 1]
```

Use the exact feature mode and EEG channel order recorded for that model. The saved scikit-learn
pipeline expects extracted features, not raw EEG. It includes the training-fitted scaler.

## Verification

The targeted suite passed 26 tests (the previously validated standalone offline W&B service test was
excluded from this run). Baseline checks include analytic sinusoid power/scaling, preservation of
within-window spectral changes, training-only standardization, validation tie-breaking, and complete
synthetic ten-fold runs for both feature modes with exact LaBraM partition matching. Real-dataset
baseline performance and cluster runtime have not yet been measured. Ruff and the CPU LSF template's
Bash syntax check passed.
