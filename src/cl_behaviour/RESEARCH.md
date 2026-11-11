# Research report: contrastive EEG-to-behaviour retrieval

**First draft | 10 November 2026 | Circling first**

## Recommendation and scope

Start with a small, calibrated, within-person retrieval problem: given EEG from a held-out Circling trial,
identify its movement among other trials from that same participant and condition. Train a shared model across
development participants, while restricting each contrastive candidate set to one participant and condition.
Keep LaBraM frozen initially. Use ordered sine/cosine movement traces, a small behavioural projection and an EEG
projection that can retain temporal structure. Evaluate both individual four-second windows and an aggregate of
their scores for the trial before introducing a learned long-trial encoder.

This is a design recommendation, not a result demonstrated by the literature. It separates four questions:

1. Does the available EEG contain recoverable movement information under the chosen protocol?
2. Is that information accessible in frozen LaBraM features, and does preserving tokens help?
3. Does contrastive training improve retrieval over a matched regression objective and the mTRF comparator?
4. Does the learned relationship transfer to held-out trials, people, conditions or datasets?

The first three should be addressed before making unseen-person retrieval the primary task.

**Evidence examined.** I read the supplied classification manuscript and Circling preprint, inspected their
relevant methods pages visually, and inspected the production model/data adapters, mTRF alignment/retrieval,
contrastive loss, token extraction, centering and trial-splitting code. The literature search emphasized original
papers and author implementations from 2024–2026, with older speech retrieval, CEBRA and BENDR as methodological
anchors. It is a targeted research review, not an exhaustive systematic review or a reproduction of those papers.
Selected public source files were inspected; their training jobs were not executed.

The reported approximately 50% mTRF retrieval and improvements from centering/full tokens are **user-reported
observations**. I did not re-run those experiments or verify their completed score artifacts in this task.
The earlier local review in `output/research/` was useful context; claims below are grounded in the sources linked
here and the current code. XAI and generative movement synthesis are outside this first implementation.

## What the existing work establishes

### Classification supplies an encoder adapter, not evidence of movement retrieval

The supplied manuscript uses LaBraM on Force Game, Mirror Game and Circling, with whole dyads/triads assigned to
train/validation/test folds. Its condition-classification results motivate testing the representation, but do not
establish that frozen embeddings preserve instantaneous movement phase. Classification may exploit sustained
spectral or visual-access differences that are insufficient for distinguishing two trials of the same condition.
Source: [classification manuscript, especially pp. 2 and 6–9](</Users/mamed/Work/LaBraM-and-XAI/Decoding_hyperscanning_EEG_using_XAI.pdf>).

The current [model adapter](../labram_classification/model.py) and
[contrastive wrapper](../hyperscanning_contrastive/overfit.py) provide the useful engineering contract:
64 ordered EEG channels, 200 Hz, 800 samples, microvolts divided by 100 exactly once, and the original pretrained
backbone. Start from that original checkpoint. A social-condition-fine-tuned checkpoint could have already seen
the proposed retrieval evaluation participants; using one requires a separate exposure audit and experiment.

### Circling and mTRF define the target and the historical comparison

The preprint describes 52 dyads performing 20 repetitions of three successive 20-second conditions:
Before / Vision / After, corresponding to `closed1` / `open` / `closed2`. It uses vertical finger displacement,
240 Hz movement recordings, 10 Hz low-pass filtering, and phase-derived predictors. Its mTRFs use lags from
−250 to +300 ms, separate models per participant/condition, and nested validation. The preprint describes pairwise
artifact rejection and excludes the dyad with reconstructed triggers from the mTRF investigation.
Source: [Circling preprint, pp. 4 and 10–12](</Users/mamed/Downloads/2026.07.03.736342v1.full.pdf>).

The executable local contracts are more specific than that methods summary:

| Item | Verified local implementation | Consequence for the new project |
|---|---|---|
| Analysis interval | First two seconds removed; movement has 4,320 samples at 240 Hz | Preserve original timestamps and the interval `[2,20)`; do not compress rejected time |
| Practice | Six EEG epochs removed, twelve for pair 2030 | Preserve source-epoch IDs and verify per-condition alignment |
| Pair 2450 | Movement has 21 trials; delete zero-based trial 10 in each condition | Apply this correction before indexing the aligned 20 trials |
| Pair 2020 | Hash-checked recovered movement onsets; retained in detailed outputs, excluded from group mTRF summaries | Keep a specific policy and sensitivity result; do not silently reintroduce it in the main comparison |
| Full target | `cos(self), sin(self), cos(other), sin(other), cos(self−other)` | This baseline is joint-record retrieval, not partner-only decoding |
| mTRF selection | Original five outer / four inner folds, seed 123, saved penalties | Recover identities from saved folds, rather than reproducing only their nominal proportions |
| Retrieval score | Candidate movement predicts EEG; mean channel-wise residual R² ranks candidates | Do not relabel this as Pearson correlation or square a correlation to reproduce it |
| Gallery | All held-out trials in the same person/condition/outer fold; K is 3 or 4 | Compare exact galleries and report query-specific chance |

Sources: [mTRF documentation](../mTRF/README.md), [alignment code](../mTRF/eeg.py),
[movement code](../mTRF/movement.py), [retrieval code](../mTRF/retrieval.py).
The actual package spelling is `mTRF`, which matters on Linux.

There is a substantive artifact-policy fork. `mTRF.eeg.retained_trials` intentionally reproduces a notebook's
`((i−1) % 20)+1` membership test against zero-based trial indices. The contrastive data reader instead interprets
the six-second dictionary with `trial = i // 3`, `section = i % 3`, and rejects overlapping intervals symmetrically.
These cannot be substituted for one another while claiming an exact baseline comparison.
Sources: [mTRF exclusion function](../mTRF/eeg.py), [contrastive interval planner](../hyperscanning_contrastive/data.py).

Keep **historical mTRF compatibility** and **verified clean-window development** as explicit profiles. For the latter,
audit movement/EEG identity and timing directly and quantify the disagreement between exclusion policies. A fair
new-protocol mTRF comparison requires refitting mTRF on that protocol; the old saved penalties do not automatically
belong to a new split. Conversely, do not “fix” the replication implementation to suit the new project.

### Reuse the contrastive engineering, reconsider its scientific assumptions

The adjacent project already has symmetric masked InfoNCE, paired true/shuffled training schedules, validation-only
selection, fixed galleries, feature provenance, geometry diagnostics and training-only centering. These are good
starting points. Its EEG-to-EEG task used a shared head; EEG and behaviour need **separate input adapters and heads**.
Its bidirectional summary is also insufficient as the primary metric here: report EEG→behaviour separately.

The [token extractor](../hyperscanning_contrastive/partner_patches.py) captures the pre-`fc_norm` grid
`[64 channels, 4 time patches, 200 features]`, discards CLS and verifies reconstruction of the original pooled output.
Keep this check. Preserve channel-major/time-minor ordering and state exactly where frozen normalization is applied.
The [video implementation](../hyperscanning_contrastive/video_projection.py) supplies compact token/channel mixing
and training-only centering examples. Its matched-movie-time centering procedure cannot be copied literally:
different Circling trials do not share the same movement stimulus across people.

## Research evidence and practical lessons

Published results below concern different modalities, galleries, repetitions and calibration regimes. They are
not a performance leaderboard, and their reported accuracies do not predict Circling accuracy.

| Work and status | Evidence relevant here | What to borrow, and what not to infer |
|---|---|---|
| **Défossez et al., speech retrieval, Nature Machine Intelligence 2023** | Contrastive alignment of M/EEG to pretrained speech representations; continuous segments rather than only image labels | Strong template for aligned temporal candidates. Speech representations and MEG results are not interchangeable with finger phase and scalp EEG. [Author implementation](https://github.com/facebookresearch/brainmagick) |
| **CEBRA, Schneider et al., Nature 2023** | Contrastive neural representations guided by behaviour, time, or both, demonstrated on calcium/electrophysiological recordings | Behaviour defines useful similarity and sampling. CEBRA-Behaviour is not automatically a two-encoder EEG→candidate retrieval system; invasive/neural-population results are not a scalp-EEG guarantee. [Paper](https://www.nature.com/articles/s41586-023-06031-6), [official usage](https://cebra.ai/docs/usage.html) |
| **BENDR, Kostas et al., 2021** | Masked contrastive learning of EEG sequences and downstream adaptation | Temporal representations and adaptation are useful precedents. Same-modality SSL success does not prove EEG–movement correspondence. [Paper](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.653659/full) |
| **LaBraM, Jiang et al., ICLR 2024** | Masked neural-token prediction; downstream tasks include gait prediction | Pretraining and channel/time tokens motivate this test. LaBraM itself is not pretrained with an EEG–behaviour contrastive loss. [Official implementation and paper link](https://github.com/935963004/LaBraM) |
| **NICE, Song et al., ICLR 2024** | EEG/image alignment with a compact temporal-spatial encoder and projection heads | Useful small-head, normalized-similarity and symmetric-loss reference. Image concepts and repeated stimulus responses differ from near-periodic movement trials. [Official code](https://github.com/eeyhsong/NICE-EEG) |
| **ATM, Li et al., NeurIPS 2024** | Maps EEG into a pretrained visual embedding space for retrieval and reconstruction | Separate alignment quality from any generator; retain temporal/spatial structure. A powerful image target encoder has no obvious counterpart for this low-dimensional movement signal. [Paper](https://papers.nips.cc/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf), [code](https://github.com/ncclab-sustech/EEG_Image_decode) |
| **EEGMirror, Liu et al., ICCV 2025** | EEG self-supervision followed by alignment to video-related semantic/dynamic targets | Staging representation learning and cross-modal alignment is relevant. Its released alignment recipe is per subject; montage flexibility does not establish unseen-person retrieval. [Official implementation](https://github.com/XuanhaoLiu/EEGMirror) |
| **NeuroDyGait, 2025 preprint / Neural Networks 2026** | EEG–motion dual encoders, motion reconstruction/prediction and relative contrastive training; subject-held-out gait evaluation | Closest executed-movement precedent found. Supports temporal behavioural encoders and a regression comparator. It predicts gait trajectories, not Circling trial identity, and does not establish LaBraM superiority. [Full manuscript](https://arxiv.org/html/2506.22488v2), [journal record](https://www.sciencedirect.com/science/article/abs/pii/S0893608026000705), [code](https://github.com/FuXi1999/NeuroDyGait) |
| **EEG2MOTION, Peng et al., August 2026 preprint** | EEG aligned with motion/video/text representations, then motion generation | Direct cross-modal motion precedent, but the acquisition is **action observation**, not execution. It also excludes three subjects after an alignment screen. Its population and task therefore differ materially from unselected interacting dyads. [Full manuscript](https://arxiv.org/html/2608.14754v1) |
| **Truong et al., October 2026 preprint** | EEG foundation-model adaptation to naturalistic video; centered targets and soft contrastive supervision | Useful motivation for target geometry, graded similarities and pretrained/random comparisons. The reported subject-scaling onset is specific to its HBN/REVE/video experiment, not a required sample size for Circling. [Full manuscript](https://arxiv.org/html/2610.09287v1) |
| **NeuralBench, Banville et al., May 2026 white paper** | Unified evaluation across EEG tasks, datasets and model types; foundation-model advantages are task-dependent | Use matched specialist and regression baselines and identical evaluation units. A general foundation model is a hypothesis to test. [Paper](https://arxiv.org/abs/2605.08495), [official framework](https://facebookresearch.github.io/neuroai/neuralbench/index.html) |

### What checking the implementations adds

The NICE training file explicitly normalizes both projected modalities and averages the two cross-entropy directions.
That is a clear minimal alignment recipe, although its data preparation and validation protocol should not be
transplanted into Circling. [Inspected NICE training code](https://raw.githubusercontent.com/eeyhsong/NICE-EEG/main/nice_stand.py).

Brainmagick's `ClipLoss` can retain the feature×time array and score aligned sequences. Its implementation normalizes
candidates and uses a directional loss; it is **not** identical to the symmetric two-sided cosine loss proposed here.
The practical lesson is that temporal alignment can survive into the scoring function without first averaging
everything into a single state vector. [Inspected loss code](https://raw.githubusercontent.com/facebookresearch/brainmagick/main/bm/losses.py).

NeuroDyGait's public training scripts and model file make EEG/motor encoding and auxiliary prediction concrete.
The combined relative-distance/reconstruction objective adds choices that we should not inherit before a basic
alignment baseline works. Reproducing its exact loss would require a separate paper–code audit.
[Training script](https://raw.githubusercontent.com/FuXi1999/NeuroDyGait/main/train_contra.py),
[model implementation](https://raw.githubusercontent.com/FuXi1999/NeuroDyGait/main/models/contraDCN.py).

EEGMirror's released instructions distinguish self-supervised pretraining from per-subject alignment and require
prepared video targets. This is useful architectural evidence, not a ready-to-run Circling pipeline.
[Alignment recipe](https://github.com/XuanhaoLiu/EEGMirror#3-stage-3--brain-modality-alignment-per-subject).

**Overall inference:** there is credible evidence for cross-modal contrastive neural decoding and for EEG–motion
representation learning. I did not identify a verified implementation that directly solves frozen-LaBraM retrieval
of an interacting partner's Circling movement under this repository's protocol. The first experiment should test
that missing link, rather than assume it from visual-decoding success.

## Design choices for Circling

### Define “behaviour” before choosing the network

Use three explicit targets with the query participant defining the roles:

- **Self:** `[cos φ_self(t), sin φ_self(t)]`. Recommended first real-data feasibility task.
- **Full:** self and partner sine/cosine plus `cos(φ_self−φ_partner)`. Required to match the existing mTRF target.
- **Partner:** partner sine/cosine only. A later, scientifically distinct test.

Self is a plausible easier task, not a guaranteed ordering. Full can be easier than partner because self motion
provides an additional retrieval cue. Good Full performance cannot establish partner-specific information.
Open trials are the first development condition; closed conditions later test context dependence. Closed-condition
self decoding can succeed, and partner motion can remain correlated through behaviour, so neither is a universal null.

Use the already verified **vertical Hilbert phase**, with its sign/origin conventions, for the first targets.
Do not replace it silently with the geometrical angle of an X/Y trajectory. Sine/cosine avoid wrap discontinuities
and preserve phase; their temporal average can nearly vanish over complete cycles, so retain their ordered samples.
At a provisional 50 Hz, a four-second target has 200 time samples: 400 values for Self or 1,000 for Full.
This rate is a starting choice to verify against target spectra and retrieval reproduction, not an established optimum.

Convert phase to sine/cosine **before** resampling with anti-aliasing. Do not linearly interpolate wrapped angles.
For exact historical mTRF, retain the original 240 Hz target; the CL adapter can derive a separately identified
lower-rate representation. Audit near-zero analytic amplitudes and stationary segments where Hilbert phase is
unstable. A later amplitude/velocity target is an ablation, not a repair applied after seeing test errors.

| Movement representation | Advantage | Limitation | Order |
|---|---|---|---|
| Ordered sine/cosine + linear projection | Phase-preserving, interpretable, few trainable parameters | Linear head may not capture nonlinear timing variation | First |
| Small 1D temporal CNN on sine/cosine | Local timing structure and shared filters | Pooling/stride can destroy the alignment being decoded | Second, retain coarse temporal bins |
| Position/velocity and phase together | Adds amplitude and speed information | Changes the mTRF target and may exploit movement artifacts | Separate target ablation |
| Magnitude spectrogram + CNN | Describes time-varying frequency content | Discards Fourier phase; similar-speed movements can look alike | Later spectral-only comparator |
| Complex STFT or time-domain/spectral combination | Retains more timing information | Larger design space and additional normalization choices | Only after simpler targets work |

A spectrogram is therefore not the default. Trial identity may depend on exactly the phase trajectory that a
magnitude representation discards. Likewise, a pretrained human-motion model for full-body actions need not be
better than two phase channels for repeated finger circling. These are signal-design arguments, not claims of
observed Circling performance.

### Retain information without creating an unnecessarily large head

For one four-second EEG window, the raw signal has `64×800 = 51,200` values. LaBraM's full token grid also has
`64×4×200 = 51,200` values. Flattening those tokens is a representation change, **not a reduction in element count**.
The mean-pooled 200-vector compresses heavily; the temporal readout has four ordered 200-vectors (800 values).

Compare pooled and temporal readouts first using the same 64-dimensional shared retrieval space and behavioural
head. A structured full-token head should follow promptly if pooling appears limiting: shared `200→16` token
projection, a few learned channel mixtures, preservation of the four time positions, then a small output map.
This draws on the local video head without assuming that its optimal settings transfer.

A dense `51,200→128` projection alone has 6,553,600 weights. Flattening eight windows and mapping to 128 requires
52,428,800 weights before any behavioural encoder. In contrast, pooled `200→64` and temporal `800→64` maps have
12,800 and 51,200 weights respectively. Parameter counts exclude biases. The full-token benefit should be tested
without confounding it with a huge trainable head; report both information retained and parameter count.

Cache frozen pre-normalization tokens and reconstruct each readout consistently. Tokens at one nominal time
position have already attended across the whole four-second input; they are not independent instantaneous samples
or strictly causal one-second summaries. An outer aggregator cannot recover information the frozen encoder lost.

### Centering: promising, but estimate it within training

A large common vector can make normalized embeddings point in similar directions. A large norm by itself does
not explain high cosine similarity. Participant centering can reduce identity-related offsets, but this should be
checked with geometry and held-out-trial identity probes, rather than inferred solely from high cosine values.
LayerNorm across features of one sample is not the same operation as subtracting a dataset or participant mean.

Compare no centering, a global training mean, and a participant training mean. Fit statistics after the chosen
readout and before the learned projection; L2-normalize the final projection for cosine scoring. Give each trial
appropriate weight so overlapping windows do not silently dominate the global mean. For structured readouts,
retain position-specific statistics unless a deliberately shared-axis normalization is being tested.

Participant means use **training trials only**, balanced across selected trial-relative offsets. Unlike the video
task, these are not matched stimulus moments across people; they may remove genuine stable movement tendencies
as well as nuisance effects. Per-feature scaling, shrinkage and PCA are additional ablations, with variance floors
and training-only fits. Do not launch all of them at once.

For unseen people, global training centering is the uncalibrated default. Estimating a new person's mean requires
explicitly allocated calibration trials and must be reported separately. Do not estimate it from the evaluation
gallery or all test EEG. Training the behaviour encoder or its normalizer on candidate test behaviour is likewise
excluded; simply encoding candidates through frozen transformations at evaluation is legitimate retrieval.

Identity-initialized feature weighting from EEG→EEG does not create cross-modal comparability: EEG tokens and
movement traces do not initially share coordinates. Untrained cross-modal heads are diagnostics, not a meaningful
“direct cosine” baseline between unrelated feature spaces.

### Four seconds versus an 18-second trial

Four seconds is the current local input contract. The upstream LaBraM implementation uses one-second patches and
a 16-entry temporal embedding table, so four seconds is not a universal architectural maximum. Feeding 18 seconds
directly would require deliberate positional/adapter changes and validation; it is not a free drop-in change.
[Upstream architecture](https://raw.githubusercontent.com/935963004/LaBraM/main/modeling_finetune.py).

The eight-window idea is valid: on `[2,20)` use starts `2,4,6,8,10,12,14,16` seconds, each of length four seconds.
That covers the full interval with a two-second stride. Overlap gives correlated contextual views, not eight times
as many independent trials. Four non-overlapping windows cover only `[2,18)` and omit the final two seconds.

Use the following sequence to isolate why more context helps:

1. **Local learning:** match four-second EEG and behaviour at identical offsets in different trials.
2. **Trial scoring without new parameters:** average matched-offset EEG–behaviour similarities across a
   prespecified window schedule. This tests whether combining local evidence is enough.
3. **Ordered compressed concatenation:** map each EEG/behaviour window to a compact vector, concatenate eight
   vectors in time order on each branch, then apply small separate trial heads.
4. **Learned sequence model:** a small temporal CNN or recurrent/attention aggregator over the compressed vectors,
   with explicit timestamps and masks. Add only if it beats the simpler aggregate on validation.

For step 3, eight pooled EEG vectors are only 1,600 values; eight compressed 64-vectors are 512. The problematic
case is concatenating eight *full* token grids. A local loss plus a trial loss is a later option, with their relative
weight chosen on validation. Local pretraining then trial-head training is easier to diagnose initially.

Overlap counts central time points more often. Prespecify equal-window scoring as a baseline, then, if needed,
compare coverage-aware weighting on validation. For a fair context comparison hold trial IDs, candidates and clean
support fixed. Missing-window patterns must not identify the correct candidate: the initial trial benchmark should
use complete eligible trials or a common allowed-offset set for every candidate in a gallery. Report exclusions.

All preprocessing is **offline decoding**. Hilbert transforms, zero-phase filtering and windows containing future
samples do not support claims of intention forecasting. The mTRF lag range motivates timing sensitivity checks,
but not copying its sign convention into a new lagged CL adapter without a synthetic alignment test.

### Loss, negatives and the easiest meaningful task

Use separate encoders with embeddings `u_i` and `v_j`, L2-normalized in the shared space, and scores
`s_ij = u_iᵀv_j / τ`. Start with the mean of EEG→behaviour and behaviour→EEG cross-entropy over aligned diagonal
pairs. Select checkpoints using **EEG→behaviour validation MRR**, with top-1 as the principal descriptive accuracy.
MRR's chance is not 1/K: it is `H_K/K`, approximately 0.521 for K=4 and 0.611 for K=3.

Negatives should be other trials of the same person, condition and relative interval. This prevents subject,
condition or nominal elapsed time alone from solving the task. Use all eligible training trials in a stratum when
practical, and average losses over several strata per update. Increasing the number of strata or accumulating
gradients does **not** increase negatives available to each query.

Do not put overlapping segments from the same trial into the negative set. Keep both members of a dyad and all
windows/targets of an experimental repetition in one split. If multiple EEG views share an identical behaviour
target, use one view per candidate set or an explicit multi-positive objective; never call the other valid view a
negative. The first implementation can avoid this complication by training each participant's stratum separately.

Near-periodic movements create genuinely similar but distinct candidates. Begin with exact-instance InfoNCE and
measure behavioural similarity between competitors. If ambiguity limits learning, evaluate prespecified soft
targets based on training-only movement similarity. CEBRA and the recent video work motivate that extension, but
it changes what the loss rewards. Avoid early hardest-negative mining or treating similarity as a reason to remove
difficult evaluation trials.

## Baselines, controls and interpretation

The existing mTRF result is valuable because it tests the actual movement records. Small absolute encoding R²
does not preclude useful ranking: the correct candidate can consistently have a smaller residual error than its
competitors even if little total EEG variance is explained.

Include these comparisons in a bounded sequence:

| Comparison | Question it resolves |
|---|---|
| Historical Full mTRF on exactly saved galleries | Can the new model improve the existing retrospective benchmark? |
| mTRF refit on the new split/support, when different | Is a difference due to representation or changed evaluation? |
| Ridge regression from the same EEG features to the same ordered movement target, then candidate scoring | Does CL add value beyond a regularized mapping? |
| Simple spectral features and a small raw-EEG temporal model, under the same galleries | Is frozen LaBraM discarding useful information? |
| Random LaBraM with identical readout/head budget | Does pretraining help, rather than architecture or random features? |
| True versus fixed shuffled-training correspondence | Does held-out success depend on correct cross-modal pairing? |
| Pooled versus temporal versus structured full tokens | Does retaining channel/time information help at controlled capacity? |

The shuffled arm should permute whole training trials within person/condition and preserve that mapping across
all their windows. Match initialization, update schedule and candidate sets. Evaluate both arms on the true
held-out correspondence. A high training score for a fixed shuffled map can reflect memorization; the important
comparison is held-out performance. Such an arm is an engineering control, not a calibrated significance test.

Additional diagnostics: tiny-set memorization, collapsed/constant embeddings with tie-aware chance, deliberately
misaligned synthetic signals, large time shifts, time-order ablations, window-level and trial-level margins,
embedding spread/effective rank, gradient norms and parameter-update norms. For periodic movement, a circular
shift near a full cycle may preserve correspondence; measure retained similarity and do not treat every shift
as a guaranteed null. Jointly shuffling both modalities' time order is different from destroying their alignment.

For scientific claims, use dyad-level paired differences and uncertainty, not windows as independent observations.
Average windows within trials, trials within participant, participants within dyad, then dyads. Bootstrap dyads
with all their members/windows/seeds retained; distinguish seed variability from population uncertainty. A formal
permutation analysis must specify an exchangeable trial/block null, preserve temporal structure and rerun learned
steps/model selection when the null requires it. One shuffled training run does not supply that inference.

Even positive self or Full retrieval can exploit residual movement/ocular artifacts or stable motor rhythm.
Include a compact raw/spectral comparator, available artifact-channel diagnostics, timing perturbations and
later speed-matched galleries before claiming specifically neural or partner-related information. Partner success
also needs comparison against prediction from self movement alone, because the partners' movements correlate.

## Recommended decisions and open questions

**Recommended now:** open-only development; Self feasibility then Full benchmark; frozen original LaBraM;
pooled/temporal small heads; sine/cosine at verified timing; within-person negatives; whole-trial splits; global
versus training-only participant centering; fixed true/shuffled arms; trial-score aggregation before sequence models.

**Defer until evidence supports them:** spectrogram CNNs, soft contrastive labels, extensive hyperparameter sweeps,
partial backbone adaptation, direct long-input LaBraM, adversarial subject removal and unseen-person evaluation.
Frozen-feature failure is evidence about this readout, not proof that behaviour is absent from EEG. Partial
adaptation or a compact raw-EEG encoder is a sensible next diagnostic if the linear baselines show signal.

Before implementation, resolve these through the Stage 0 audit and a short design discussion:

1. Which completed mTRF retrieval directory is the authoritative baseline, and can its saved fold/gallery artifacts
   be accessed? It is not identified in the request. Do not regenerate a different gallery and call it identical.
2. Which dyads/trials remain eligible for a genuinely new final evaluation? Existing mTRF summaries already informed
   this project, and prior condition-classification/EEG retrieval may have exposed additional data. Record an exposure
   ledger; a new random split does not erase earlier inspection.
3. Is the primary eventual scientific claim joint movement identification, self movement decoding, or information
   about the partner beyond self motion? The first draft supports all three as separate targets.
4. What is the agreed new-project artifact policy after comparing notebook exclusions and six-second interval
   masks against source metadata? Preserve historical reproduction separately regardless of that decision.
5. Are calibrated known-person use and later unseen-person use both desired? Participant centering is compatible
   with the first, but the second needs a separately budgeted calibration experiment.

The [implementation plan](PLAN.md) turns these recommendations into small experiments with explicit stopping rules.
