# Shared-stimulus video EEG: proposed experiment plan

Status: Stage 0 audit/preparation implemented and locally audited, 23 October 2026. Training remains planned.
Local source: `/Users/mamed/Work/data_for_mario 2`.

## Objective and interpretation

Primary engineering question: can EEG from person A identify the matching four-second movie interval in person B,
among other intervals from that same movie? This targets stimulus-locked information across people. It does not
require co-presence, and success in solo viewers is expected to be possible. It does not by itself establish
engagement, social coupling, emotional processing, or a hearing-impairment effect.

Later scientific questions: does retrieval vary with low/high babble and solo/together presentation? Does an actual
co-viewing partner show excess similarity beyond other viewers of the same movie moment? Distinguish these from
basic stimulus decoding. Cross-video distractors may make retrieval easier through movie identity; held-out-video
generalization is the genuinely stronger test. Do not equate larger galleries with automatically better learning.

## Verified inputs and remaining uncertainty

The user confirms 80 separate people and synchronized, uninterrupted exported segments. There are four babble
files, not always shared between viewers. Exact stimuli, timing logs, ratings and colleague analysis outputs are
not currently available. The local README describes noise levels but does not provide babble-file assignments.
Record these assignments as unknown; A/B is counterbalance, not babble identity or a fixed noise level.

Read-only inspection found 714 NPZ files: 354 solo and 360 together, 40 people per presentation condition.
Together comprises 20 pairs. All inspected headers have a common 64-channel order. Six absent files match the
colleague's description. Video0 is approximately112 seconds, not140–195 seconds as the broad README estimate
suggests; other films range approximately142–196 seconds. Header-based capacity is28,916 nonoverlapping4s
person-windows before exclusions, common-duration clipping, edge guards, or split purging. This is not the number
of independent observations or distinct stimulus moments. Across nine films there are only roughly360 distinct4s
positions; pair combinations reuse those same responses.

Exclude solo11_A_1 as instructed by the colleague. Quarantine pair4_together_A_4_p2 and pair9_together_A_9_p2 as
provisional quality exclusions, separately from the confirmed faulty recording. This leaves77 people and688
files, with27,859 header-derived windows before other guards. Retain usable p1 recordings for cross-person
analysis; omit incomplete pairs from analyses requiring both actual partners. Preserve pair grouping regardless.

All180 together partner-video pairs have equal exported lengths, but different original sample indices. Checked
examples have a constant28-sample p2 delay (~13.7ms at2048Hz). Same-film duration ranges across viewers span
approximately0.17–0.21s. Given the user's synchronization confirmation, use segment-relative time zero and retain
original indices as provenance, not as timestamps shared across recording sessions. Do not invent corrections,
align by maximizing EEG correlations, or stretch recordings to equal duration. Record alignment provenance as
reported photodiode correction, not independently verified frame alignment. Tail differences motivate clipping
and guards, not rescaling. Optional later lag-sensitivity diagnostics must select lags on training data only.

Three inspected signals had finite values and plausible amplitudes (~5–7 microvolt standard deviations); this is
a spot check, not an all-file signal-quality audit. The processing stage must inspect every file.

## Data-processing contract

- Use labelled NPZ files; never infer video IDs from positions in pickled lists.
- Parse participant, condition, together-pair ID, member, counterbalance, and session/config index independently.
- Verify explicit video IDs, sample rate, shape(time,64), finite values, channel uniqueness, trigger-duration
  consistency, duplicate recordings and partner completeness. Record missing/excluded files explicitly.
- Map noise per video: video0=no noise; videos3–6 low inA/high inB; videos7–10 high inA/low inB.
- Preserve cleaned data; do not repeat ICA, interpolate again, or import Circling's artifact dictionary.
- Reorder channels by names to the existing LaBraM montage. Input units are volts; store microvolts exactly once.
- Resample continuous film segments256→200Hz with anti-alias filtering, then cut exact800-sample windows.
  Preserve native256Hz access for CorrCA; its representation should retain within-window time structure.
- Use a stimulus-relative4s grid. Reserve configurable edge guards, initially at least4s at each film end.
  Derive one common valid time support per film conservatively; log the discarded tails. Never pad missing EEG.
- Flag high amplitude, flatness, saturation, excessive discontinuities and nonfinite values. Fix quality policies
  before scoring. Removing a window must never concatenate time or change subsequent stimulus timestamps.
- Save one signal array per participant/video plus a window manifest; form pairs dynamically rather than
  materializing all combinations. A bad person-window invalidates pairs involving it, not every other viewer.
- Manifest fields: source hash/path, participant_id, independence_group, pair_id/member, video_id, condition,
  counterbalance, noise_level, babble_id(nullable), source256Hz indices, resampled200Hz indices, time_start/end,
  QC reason, eligibility and split. Save inclusion policy and preprocessing version in cache metadata.
- Produce census by film, noise, social condition, subject, pair, and split; report unique moments, person-windows,
  valid partners, candidate counts, and effective updates separately.

## Split and gallery contract

Before experiments, reserve participant groups for later evaluation, stratified by social condition and A/B.
Together partners always share a group; solo participants are separate groups. Exact fold sizes follow QC.

Initial development uses seen people but held-out contiguous movie-time blocks. A given video-time block belongs
to the same split for EVERY participant: holding out only an A–B pairing while training A–C on the same moment
would not test unseen content. Purge boundary windows with at least a4s gap, checking this against known filter
support where available. Existing zero-phase preprocessing has finite support that cannot be fully audited without
its code; document this limitation and use a wider-gap sensitivity analysis later. Random interleaved window splits
are inappropriate. Save split manifests once; do not regenerate for each model.

Use fixed K=4 galleries for the smallest no-noise held-time pilot, increasing toK=8 when enough separated held-out
positions exist. Do not silently borrow training positions to fill validation galleries. Keep distractors in the same
video and target person; use at least8s onset spacing initially to avoid immediately adjacent4s windows. Report
exact candidate counts and chance values. Later evaluate all eligible positions, including nearby distractors,
as an explicitly harder temporal-localization task. Subject-specific scores should not be dominated by the number
of possible partners. Aggregate equally over target/query identities and films; account for dyad and film dependence.

## Batch construction

First train with two people andK unique common valid times from one movie/noise/social stratum. Each side hasK
windows; the K×K cross-person similarity matrix has correct aligned pairs on its diagonal. Use symmetric InfoNCE.
Start K=8 or16 according to the training census; two people×16 times means32 encoder inputs,16 candidates per
query and15 negatives, NOT32 negatives. Accumulating independent losses does not increase this candidate count.
Rotate participant pairings across updates and avoid same-session co-viewing partners in the first general
shared-stimulus experiment. Sampling with replacement does not create unique negatives. Log coverage each epoch.

Next, multi-person batches can use M people×K times. All other people's copies of the same time are positives.
Use a multi-positive loss (or average all cross-person K×K losses); never label matching-time third viewers as
negatives. M=4,K=16 provides64 inputs,3 positives per query in an all-other-person formulation, and45 eligible
other-time cross-person negatives before any additional temporal masks. Counts must be reported explicitly.
Do not feed video IDs, timestamps, participant IDs or condition labels into the encoder; these define sampling only.

## Stages and decision rules

0. Audit and prepare: validate timing/units/channel/QC contracts; produce metadata and signal census. No model fits.

1. Synchrony and engineering controls: small synthetic aligned/misaligned EEG controls, tiny-set fitting, and
training-only regularized CorrCA on native signals. Project held-out times and compare aligned versus shifted
responses. Start with solo video0 (no babble, different recording sessions), then solo low-noise films. Use longer
8/16s correlation estimates as diagnostics if4s is too noisy; keep4s retrieval as the LaBraM-compatible primary task.
Video0 is always first and is only one movie, so it is a pilot, not evidence of a no-noise condition effect.

2. Cross-person, held-time retrieval: compare raw standardized waveform correlation, training-fitted CorrCA
projected waveform retrieval, and frozen pretrained LaBraM pooled readout using identical galleries. CorrCA is
correlated-components analysis with shared filters, not pair-specific canonical correlation. Fit filters,
regularization and any feature scaling only on training people/time blocks. Add a small spatial-temporal CNN
inspired by CL-SSTER early; frozen LaBraM failure must not veto a promising dataset. Use paired true and fixed
misaligned training controls, initial/final/best curves, multiple seeds, and per-person/per-film results. Progress
requires a reproducible advantage on unseen time blocks, beyond both initialization and misaligned controls.

3. More viewers: expand from solo low noise to matched strata including together (initially across different
pairs), then multi-positive batches. Match noise level first; retain unknown babble identity. Separately evaluate
cross-noise/counterbalance transfer, which is not an identical-acoustic-stimulus objective.

4. Unseen people: evaluate frozen settings on reserved groups. Distinguish new people seeing training moments
from new people seeing held-out moments; do not conflate participant transfer and stimulus-content transfer.
Predefine whether both query/gallery identities are held out or gallery uses a fixed training reference set.

5. Unseen films: outer held-out-film experiments with all model selection inside training films. This tests
transfer of shared processing rather than memorization of particular movie moments. Nine films limit precision.

6. Scientific comparisons: social condition×noise effects, and real-partner excess versus same-moment different-
pair viewers. Noise is within-person across films, social condition between people, and counterbalancing must be
respected. Within a person noise is also tied to different films, so model film effects explicitly. Shared babble,
shared recording hardware, and group allocation can confound interpretation. Engagement requires convergent
behavioral measures; without them use the term shared-stimulus retrieval. Missing babble assignments limit nuisance
controls; missing exact movie files prevent reliable stimulus-feature regressions using generic YouTube copies.

## Controls and model choices

Use frozen pooled LaBraM and the small linear head first; no large patch-head sweep or attention expansion.
Initialize a fixed starting setting (temperature0.5, head LR1e-4) based on the existing implementation, explicitly
not a proven optimum for this dataset. Keep a random-backbone comparator after the first positive result.
Add limited encoder adaptation only after the simpler cross-person objective works reliably.

Misaligned-training controls must preserve subject/video/noise composition. Validation circular shifts or block
permutations should break shared moment identity while preserving within-person temporal structure, avoiding
near-zero shifts and artificial wraparound windows. A single shuffled-trained model is an engineering control,
not a null distribution for significance. Confirmatory inference must account for repeated model selection and
shared subjects, pairs and films; millions of pairs are not millions of independent samples.

## Repository integration and immediate next deliverable

Keep this under Project2 `hyperscanning_contrastive`, with a dataset-specific video loader/manifest/sampler and
separate commands/cache format. Reuse LaBraM loading, losses, metrics and plotting; do not force multi-viewer data
into the Circling dyad-only cache or change existing Circling commands. Implemented commands: video-audit and
video-prepare, with the CPU hpc/video_data.lsf wrapper and synthetic fixture tests. See README for usage.
First deliverable is the audit/census and cache with no training, followed by a small same-video held-time pilot.
Cluster source path remains to be provided at implementation time.

## Primary references

- Cohen & Parra (2016), Memorable Audiovisual Narratives Synchronize Sensory and Supramodal Neural Responses,
  supplied PDF ENEURO.0203-16.2016. The paper studies audiovisual processing/memory, not this noise/co-presence
  factorial design. Its movie-level ISC result does not guarantee4s retrieval success.
- Shen et al. (2024), Contrastive learning of shared spatiotemporal EEG representations across individuals for
  naturalistic neuroscience, NeuroImage120890: https://doi.org/10.1016/j.neuroimage.2024.120890
  Direct methodological precedent: aligned cross-person segments and spatial-temporal convolutional features.
  Reported ISC improvements are not retrieval accuracy forecasts for this dataset.
- Official CL-SSTER implementation: https://github.com/kekehia123/cl_sster


## Stage 0 local audit outcome

All 714 files scanned. Default exclusions retain 77 participants. Of 28,916 complete native four-second windows,
26,167 pass inclusion, common support/edge guards, and gross artifact screening, covering 345 stimulus moments.
Eligible windows: 13,119 solo and 13,048 together. Screening thresholds are explicit configurable defaults,
not a replacement for a validated artifact detector. The six absent recordings match the supplied description.
No nonfinite recordings or exact duplicate signals were found. Splits remain unassigned.

## Stage 1 implementation

`video-baseline` now implements the solo/no-noise held-time waveform/CorrCA pilot and misaligned-training control.
It uses the portable verified 200Hz cache rather than reopening native 256Hz recordings, retaining all temporal
samples. Tiny neural-network fitting and 8/16s sensitivity analyses remain later checks. See README for the exact
participant reservation, temporal split, galleries, covariance weighting, outputs, and limitations.

## Frozen representation comparison

`video-labram` now reuses a completed `video-baseline` run's exact gallery and split manifests. It extracts
original pooled frozen LaBraM features and compares cosine retrieval with saved waveform/CorrCA scores.
No training, centering, projection, new splits or reserved-person evaluation occurs. See README and the CPU
`hpc/video_labram.lsf` wrapper. This implements the frozen-LaBraM comparison portion of Stage 2; contrastive
training and the small spatial-temporal CNN are still subsequent steps.

## CNN pilot implementation

`video-cnn` implements the compact task-trained comparison after the frozen pooled LaBraM pilot. It preserves
saved identity/time splits and validation galleries. Training uses two viewers and K=8 common moments; unlike
validation, adjacent nonoverlapping training moments are allowed. The matched control rotates selected target
windows per update, preserving exact inputs but creating stochastic false pairings rather than a global fixed
mapping. A disposable tiny-set fit precedes both fresh-initialized arms. See README for architecture, scale fitting,
optimization, checkpoint selection and coverage outputs. Multi-seed confirmation and wider held-time blocks remain
necessary before interpreting improvements as reliable generalization.
