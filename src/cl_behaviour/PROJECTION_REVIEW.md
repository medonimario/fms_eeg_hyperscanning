# Projection review after sweep 30021280

Discussion proposal, not an implemented or approved architecture change. The user's temperature-0.5 sweep is
still running. No result from that sweep is assumed here.

## What the experiments establish

The full EEG map has 51,200 × 32 = 1,638,400 weights, plus the movement map's 400 × 32 = 12,800.
It fits almost perfectly across all six LR/decay settings but generalizes weakly. Temporal's 25,600 EEG weights
plus the same movement head are more successful, with approximately .3596 selected validation MRR versus .3437
chance, but this is a small single-seed validation-selected advantage. Tiny-fit success establishes fitting,
not useful movement information in the encoder. Correlated windows are not independent data: 7,616 training
participant-windows originate from 952 paired training trials across 41 dyads.

The evidence motivates testing spatial structure at comparable capacity to temporal. It does not establish
that head capacity is the only bottleneck. Both modality heads are trainable; a dense EEG map can fit arbitrary
training correspondences, as the shuffled loss already demonstrates.

## Origin and exact meaning of a constrained projection

The original RESEARCH.md proposed shared feature projections and channel mixtures. The existing
hyperscanning_contrastive/video_projection.py VideoHead compact/full implementation uses a shared 200→16
map with bias/GELU, a 64→3 signed channel mixture and a final 192→64 map. That is a local engineering precedent,
not evidence of success for EEG-to-movement. Any new implementation would stay inside cl_behaviour.

One specifically motivated new comparison is a *linear* structured head:

- Keep the full normalized/centered grid X with shape channel 64 × time 4 × feature 200.
- Apply A: 200→32, shared across every channel/time token (6,400 weights).
- Apply S: 64→4 signed channel mixtures, shared across time and feature (256 weights).
- Preserve the four time slots; flatten 4 mixtures × 4 times × 32 features = 512.
- Apply U: 512→32 (16,384 weights), then the existing L2 normalization.

EEG total: 23,040; both heads: 35,840, close to temporal's 38,400 and about 46× smaller than current full.
The mixture and feature widths are chosen to provide a near-capacity-matched temporal comparison, not claimed
as biologically optimal numbers. No biases, GELU, attention, new movement encoder or loss would be added.

Before final L2 normalization this remains linear, with effective coefficients
W[o,c,t,f] = sum_(r,h) U[o,r,t,h] S[r,c] A[h,f].
Thus spatial mixtures and feature bases must be reused. It cannot independently assign arbitrary feature bases
to each channel/time pair. It preserves time order and learns several channel combinations rather than averaging
all channels uniformly. Learned mixtures are not automatically anatomical regions; contextual tokens have
already mixed information through LaBraM attention. The hypothesis can fail if important information requires
richer channel-feature interactions or is absent from frozen tokens.

This is stronger than ordinary norm shrinkage: multiplying a bias-free head by a positive scalar does not change
its final cosine-normalized embedding. AdamW can still affect learning, but weight decay is not equivalent to
removing free spatial degrees of freedom. A generic linear bottleneck of width >=32 would not constrain the
rank beyond the existing 32-dimensional output; the proposed constraint is on spatial/feature structure.

## Relevant research and limits

- EEGNet: compact depthwise/separable EEG operations provide a precedent for parameter sharing and structured
  spatial/temporal processing. This is raw-EEG classification, not proof for frozen LaBraM movement retrieval.
  https://arxiv.org/abs/1611.08024
- Decoding speech perception from non-invasive brain recordings: spatial attention and participant-specific
  processing accompany a shared contrastive architecture. This motivates testing person-specific mappings;
  subtracting a participant mean does not align participant-specific rotations or delays.
  https://www.nature.com/articles/s42256-023-00714-5
  https://github.com/facebookresearch/brainmagick
- EEG-to-Gait Decoding via Phase-Aware Representation Learning (NeuroDyGait): combines EEG/motor objectives with
  reconstruction/prediction and session-specific heads. Its gait task, learned EEG encoder, and evaluation differ
  from Circling. It motivates motor-target and sharing diagnostics, not copying its whole architecture.
  https://arxiv.org/html/2506.22488v2
- Truong et al.: soft-target supervision comes from centered frozen video-teacher similarities. Their teacher
  and partly adapted EEG system differ from two freely fitted heads on frozen LaBraM. Raising temperature .1→.5
  softens predictions/changes gradients but does not change our one-hot target matrix or implement their method.
  https://arxiv.org/html/2610.09287v1

## A small hypothesis-driven next block

First review the already-running temperature sweep. Compare validation ranks, not absolute loss across
temperatures. Retain the locked test sets. Choose and record a reference configuration before new comparisons.

Diagnostic priority: on the same temporal features, movement vectors, offsets, identities and galleries,
compare shared ridge prediction with participant-specific ridge prediction, with a small prespecified common
regularization grid selected by macro validation rather than independently tuning each person. Rank movement
candidates using a fixed similarity to the predicted waveform. This tests fixed-target accessibility and whether
sharing is an obstacle. It is not an isolated loss ablation because target space and fitted capacity also change.
Do not claim that failure proves the features contain no information.

If a head comparison is pursued, compare one prespecified structured head above with the existing temporal
reference, leaving movement head, loss, normalization, schedules and galleries fixed. Use matched shuffled arms
and three prespecified seeds for both heads; no sweep of mixtures/widths at this stage. Report per-dyad differences,
train/validation gaps and seed variation, not only the best checkpoint number.

Interpretation:
- Structured improves validation reproducibly: spatial information may be useful when constrained; retain it.
- Structured fits less but does not improve validation: reduced memorization alone is not success.
- Participant ridge substantially beats shared ridge: prioritize participant-aware mapping before bigger heads.
- Fixed-target regression works but CL is weak: investigate learned target geometry and contrastive supervision.
- Both ridge probes are weak: compare a raw-EEG mTRF control on the *same* windows/splits/galleries before blaming
  the CL loss. Existing ~50% mTRF is 18-second, different candidate/task/model-sharing conditions, and its local
  retrieval code predicts EEG from each movement candidate; it is not directly comparable to current results.

Movement similarity analysis should precede soft targets: verify whether different-trial candidates are near
duplicates under a fixed phase-aware representation, and whether learned movement embeddings discard differences.
Current galleries contain equal-offset windows from different trials, not adjacent windows of one continuous
trial. Equal offset must not become a temporal-soft-positive rule across trials.
