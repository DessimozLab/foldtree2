# SE(3) reconstruction: diagnostic plan, 2026-10-20

## Goal and current evidence

Recover complete single-chain CA folds reliably, rather than merely reducing
the training objective. Full-backbone and atom reconstruction are later goals.
The [growth trial](se3_growth_trial_20261014.md) reached mean TM-score 0.215,
CA RMSD 15.69 Angstrom and CA lDDT 0.255 on the monitored development set.
[Continuation](se3_continuation_20261014.md) showed diminishing structural gains.
The [loss comparison](se3_loss_structure_20261014.md) remains pending in the
available local record; queued status is not evidence of completed learning.

Production weights and graph inputs remain unchanged. Do not scale data,
capacity or duration automatically. Every subsequent experiment requires a
diagnosis from the preceding stage.

## Stage 1: reconstruction capability controls

Use the verified local 4,096-training frozen bundle. Select eight training IDs
deterministically across stored lengths 64--128, requiring at least 90% valid
residues. This is a length-spread diagnostic panel, not a fold-class-balanced
or independent test set. Record IDs, masks through cache identity, calibration,
source snapshots, software, GPU, initial/final paired metrics and checkpoints.
No production encoder or decoder is called.
Filter low-confidence candidates before selecting the length-spread panel;
preserve all excluded IDs and their valid fractions in `selection.json`.

Two independent local GPU queues:

1. GPU 0: directly optimize coordinates from the normal calibrated/lifted seed;
   then repeat from native coordinates plus Gaussian noise (1 Angstrom standard
   deviation per Cartesian axis). Up to 2,000 updates / 30 minutes per case.
2. GPU 1: fit the normalized width-64/depth-4 refiner on one training structure;
   then fit the eight-chain panel from a fresh initialization. Up to 2,000
   updates / 30 minutes per case. Each update averages all panel structures;
   the one-chain and eight-chain cases have different training-visit counts.

All cases use the combined coordinate loss: distance, soft FAPE, bonds,
nearby nonlocal distances, bends and signed torsions. Direct optimization
disables the auxiliary angle loss, since there is no learned angle head.
Model training retains its auxiliary angle loss. Learning rates are 0.05 for
direct coordinates and 0.0005 for model fitting, with gradient clipping at 1.
The model uses the existing normalization, residual gain, calibration and
normal lift. Model parameters/optimizer are float32; serial panel gradients
accumulate in float32, without making any cross-topology equivalence claim.
There is no scheduler or early success stop. Walltime is checked at complete
update boundaries, so startup/evaluation/checkpoint overhead is additional.

Native-noise inputs and freely optimized coordinates are **oracle diagnostics**,
not deployment paths. The model's actual-input memorization cases consume no
target-derived input coordinates. All use training targets for supervision.
Checkpoints are diagnostic snapshots, explicitly not resumable training jobs.
Best snapshots are selected by each case's own training objective at logged
evaluation points; do not compare objective magnitude across different losses.

Record TM-score, aligned RMSD, CA lDDT, bonds, bends, torsions, original FAPE,
unclamped frame error, FAPE saturation and long-range contact precision/recall
(CA distance below 8 Angstrom, row separation at least 12). Undefined contact
ratios are null with explicit counts. Existing metrics/masks remain unchanged.
Provisional small-panel reconstruction targets are TM-score above 0.8 and
aligned CA RMSD below 3 Angstrom **per chain**, together with improving local
geometry. A 20% objective decrease alone does not pass reconstruction.

Interpretation:

- Direct native-noise recovery fails: investigate loss/frame conditioning and
  direct optimization before blaming representation capacity.
- Direct seed optimization fails but native-noise succeeds: investigate the
  global objective landscape and seed, not just local geometry weights.
- Direct seed optimization succeeds but one-chain model fitting fails:
  investigate network parameterization, gradients and coordinate updates.
- One-chain fitting succeeds but panel fitting fails: investigate capacity,
  conflicting gradients and conditioning before increasing dataset size.
- Memorization succeeds: proceed to representation and held-out controls.

An unsuccessful bounded optimization is not proof that recovery is impossible.
Inspect curves, gradients and structures, then run a targeted control.

## Stage 2: information and seed ablations

Separate continuous latent + codebook inputs from quantized/codebook-only
inputs. Restricted-input experiments must regenerate their seeds and contacts
from the same allowed representation; otherwise continuous information leaks
through those channels. Measure long-range distance/contact prediction from
frozen features. Compare removal/shuffling of contact edges and target-free
seed alternatives under matched training budgets.

The original contact bottleneck is planar; the fixed normal lift removes the
rank constraint but does not establish a good global initialization. The
separate auxiliary angle head does not construct the backbone.

## Stage 3: evidence-directed architecture experiments

Only after the controls, choose among objective conditioning, stronger pair
conditioning, iterative coordinate refinement, alternative seed construction,
or revisiting quantization. Do not increase depth as a default response.
Retain both control and candidate metrics on the same structures.

### Compute escalation

The user authorized Clariden for experiments that need more infrastructure
than the two local GPUs. Keep bounded diagnostics local; use Clariden when
measured memory needs or experiment throughput justify it. Follow the local
`clariden` skill and the [existing cluster runbook](se3_clariden.md).
Before submission, check current access, live resource limits, project credit
and the shared budget ledger; do not infer remaining credit from old records.
Declare nodes, GPUs, partition, walltime and maximum GPUh. Preserve pinned
source/container identities, verified caches, explicit tensor normalization
and float64 gradient accumulation for the validated distributed model.
New distributed variants require bounded real-training, checkpoint recovery
and topology checks before expansion. Keep outputs durable, avoid automatic
retry loops, and do not treat cluster availability as a reason to scale before
the reconstruction gates pass.

## Stage 4: reliability validation

Reserve a new sequence/structure-cluster-separated test set. Audit production
pretraining overlap where possible; otherwise state the limitation. Use the
existing monitored holdout only as development data. Report per-chain success
rates, lower-tail performance and length/fold strata across multiple seeds.
The existing model constructor resets seed 42 internally; make initialization
seeding explicit and test it before claiming multi-seed validation.

## Execution

Run [the diagnostic runner](../scripts/se3/diagnose_se3_reconstruction.py) in the
CUDA-capable `foldtree2` environment. Example:

```bash
python -m scripts.diagnose_se3_reconstruction \
  --mode coordinates --device cuda:0 --learning-rate .05 \
  --output runs/se3_reconstruction_20261020/coordinates_seed
python -m scripts.diagnose_se3_reconstruction \
  --mode model --device cuda:1 --count 1 --learning-rate .0005 \
  --output runs/se3_reconstruction_20261020/model_one
```

Use fresh output paths. Failures are logged and re-raised; later queued cases
must not run after a failure. No automatic retries or production promotion.

## Local execution status

Both RTX PRO 4000 GPUs were idle before launch. Focused diagnostic/loss/refiner
tests passed (45 tests), and documentation checks passed (56 documents,
65 CLI examples). Real-cache CUDA smoke checks passed for direct seed
optimization, direct native-noise optimization and eight-chain model fitting.
An earlier smoke attempt stopped before optimization because confidence was
checked after selection; selection now filters candidates first. The failed
attempts remain preserved, without changing confidence thresholds or masks.
The panel was selected from 1,029 eligible chains; 538 length-eligible chains
were excluded for valid-residue fraction below 90%.

Persistent user services were launched and verified active on both GPUs:

| GPU | Service | Sequential cases |
|---|---|---|
| 0 | `foldtree2-se3-coordinates-20261020.service` | `coordinates_seed`, `coordinates_native_noise` |
| 1 | `foldtree2-se3-memorization-20261020.service` | `model_one`, `model_panel` |

Each service has an additional 65-minute hard ceiling. Outputs are under
`runs/se3_reconstruction_20261020`; queue logs are `gpu0.log` and `gpu1.log`.
Each case preserves provenance, selection/exclusion records, initial metrics,
events, best/last snapshots and paired metrics, and a completion summary.
An active service or a passed smoke test does not establish fold recovery.
Completed comparisons must be reviewed before Stage 2.

```bash
systemctl --user status foldtree2-se3-coordinates-20261020.service \
  foldtree2-se3-memorization-20261020.service --no-pager
tail -n 5 runs/se3_reconstruction_20261020/gpu0.log
tail -n 5 runs/se3_reconstruction_20261020/gpu1.log
```

## Expanded controls authorized on 2026-10-20

Keep the original runs intact. Two follow-up queues wait for their respective
original service to finish successfully, require both parent completion records
without failure events, and then execute sequentially:

| GPU | Case | Purpose |
|---|---|---|
| 0 | `coordinates_control` | Matched direct optimization with legacy soft-FAPE/distance/bond loss, without the new coordinate terms |
| 0 | `model_native_noise` | Eight-chain model fit from fixed noisy native coordinates; oracle local-recovery control |
| 1 | `model_panel_transfer` | Evaluate the original panel's training-selected best checkpoint on 16 development holdouts, without optimizer updates |
| 1 | `model_control` | Matched eight-chain memorization with legacy loss |
| 1 | `model_local_contacts` | Matched eight-chain memorization with only adjacent-row edges, removing both inferred nonlocal and seed-distance contacts |

All training cases retain the original 2,000-update/30-minute bounds,
initialization seed, architecture, training IDs, confidence filter, clipping
and learning rates. Native-noise model inputs are fixed per chain, not
freshly resampled denoising examples; their success does not establish a
general denoiser. The local-contact ablation retains features and the
calibrated/lifted seed, so it does not remove all upstream structural
information. Native-noise and direct-coordinate results are oracle controls,
never evidence of target-free reconstruction.

The new model cases evaluate the same 16 length-spread, high-confidence
development holdouts before training and at the last and training-selected best
checkpoint. They never backpropagate holdout losses or select checkpoints with
them. These are reused development data, not an untouched generalization test;
the native-noise holdout also uses oracle coordinates. Best-checkpoint
evaluation checks bundle, panel, architecture, loss, input settings and model
source identity against the saved checkpoint before loading. Selection and
provenance retain both training and evaluation IDs.

Every new run audits coordinate loss at perfect real target coordinates,
rejecting a weighted coordinate objective above 1e-5. Auxiliary angles are
reported separately and are not part of this perfect-coordinate audit.
CPU smoke checks verified local-only contacts, held-out monitoring and
checkpoint-only evaluation. The expanded regression suite passed 48 tests.
The queue entrypoint is
[queue_se3_reconstruction_controls.py](../scripts/se3/queue_se3_reconstruction_controls.py);
it stops on any failed parent or child, with no retry or automatic expansion.
Follow-up services have a 140-minute hard ceiling including waiting for their
parents; each queue permits at most 70 minutes of parent waiting.

These controls test the existing continuous-feature system. Discrete-only
decoding, alternative seeds, multiple initialization seeds and fresh held-out
training remain Stage 2/4 experiments; changing upstream information channels
requires a separately verified cache rather than an incomplete feature toggle.

Both follow-up services were verified active and waiting on their parents:
`foldtree2-se3-controls-gpu0-20261020.service` and
`foldtree2-se3-controls-gpu1-20261020.service`. Their logs are
`controls_gpu0.log` and `controls_gpu1.log` under the same run root.
No extra GPU workload starts while the original queue occupies that device.
No Clariden allocation was needed for this stage.

At this observation, one-chain training had completed 2,000 updates; its
training-objective-selected checkpoint had TM-score 0.9161, RMSD 1.6518 Angstrom
and lDDT 0.9302. Direct seed optimization was still running at update 1,725;
its selected checkpoint had mean TM-score 0.9845, RMSD 0.4265 Angstrom and
lDDT 0.9820, with all eight chains passing both global-shape thresholds.
The eight-chain model was only at update 150, mean TM-score 0.2989 and RMSD
10.3299 Angstrom. These panel-model and direct-coordinate values are interim,
not completed comparisons. This supports recoverability under direct
supervision, not held-out reconstruction; backbone metrics and all completed
per-chain comparisons still need review.

## Remaining test queues: nested fitting and information controls

The user authorized queuing all next-stage diagnostics. Each GPU waits for
its expanded-control service to finish with a successful service result and
clean child completion events. Waiting is capped at 150 minutes; the new
services have a four-hour total hard limit. Training jobs retain the original
2,000-update / 30-minute per-case bounds. Failure stops that GPU's queue.

| GPU | Case | Details |
|---|---|---|
| 0 | `ladder_2` | First two members of the fixed eight-chain panel |
| 0 | `audit_2`, `audit_1` | Inspect training-selected ladder and original one-chain checkpoints |
| 0 | `information_probes` | Sequence-separation baseline, codebook-only and continuous-feature probes |
| 1 | `ladder_4` | First four members of the same panel |
| 1 | `audit_4`, `audit_8` | Inspect ladder and original eight-chain checkpoints |

The existing one-chain and eight-chain runs provide the endpoints; do not
retrain them unnecessarily. All nested model runs start from the same model
initialization (seed 42), averaged-panel loss and AdamW settings. Report
completed updates and structure visits, since equal update counts do not mean
equal visits. Time limits can also produce unequal update counts. The ladder
cases additionally retain 16 development holdouts and calibrated/lifted input
baselines. They do not use holdouts for checkpoint selection.

Checkpoint audits compare train/eval predictions, objectives and structural
metrics on the same inputs. They record per-chain parameter-gradient norms,
the complete pairwise gradient-cosine matrix, and output-minus-seed coordinate
RMS. Training-mode BatchNorm buffers are restored after every chain and in
failure cleanup; gradients are obtained without populating parameter `.grad`
or taking optimizer steps. No dropout is configured. Audits measure output
displacement, not hidden layer-by-layer coordinate steps. They use saved
training-selected checkpoints, not a holdout-selected model.

Information probes use 64 high-confidence training chains and 16 development
holdouts of 64--128 stored residues. Deterministically sample 256 unordered
pairs per chain with row separation at least 12, using identical pairs for all
representations. Inputs are symmetric feature sums, absolute differences and
products, plus log sequence separation. They contain no target coordinates,
seed coordinates or contact edges. The sequence-only probe zeroes residue
features; the codebook-only probe uses quantized vectors, padded with zeros so
all three MLPs have identical input width, parameter count and initialization.
Standardization is fitted on training pairs only.

Each probe predicts positive distance and a separate contact logit. Its
objective is SmoothL1(distance/10) plus 0.1 times binary contact cross entropy,
with contact defined as CA distance below 8 Angstrom. Train up to 2,000 updates
or five minutes per representation, LR 1e-3 and batch size up to 1,024.
Report final fixed-budget training/validation distance MAE, contact
precision/recall with explicit counts, per-chain metrics, a constant
training-mean distance baseline, and completed update counts. No holdout-driven
selection or tuning. Sparse contact ratios may be undefined and remain null.
Identical initialization/width do not imply identical effective capacity
when input channels are zeroed.

These are pairwise information probes, not quantized-only fold reconstruction.
A poor small probe is not proof that information is absent. Full discrete-only
reconstruction still requires a verified regenerated seed/contact cache.
Multiple initialization seeds and fresh test-set confirmation remain gated.

The runner is
[queue_se3_remaining_controls.py](../scripts/se3/queue_se3_remaining_controls.py);
probe implementation is
[probe_se3_information.py](../scripts/se3/probe_se3_information.py).
Focused validation passed 52 tests. Real-cache CPU checks verified matched
probe execution and checkpoint train/eval/gradient auditing. No extra GPU work
is launched before its preceding queue finishes.

The remaining-control services were launched and verified active, waiting
without allocating additional GPU work:
`foldtree2-se3-remaining-gpu0-20261020.service` and
`foldtree2-se3-remaining-gpu1-20261020.service`.
Logs are `remaining_gpu0.log` and `remaining_gpu1.log` in the existing run
directory. Newly started reconstruction runs retain per-chain metrics in each
progress event, in addition to mean curves and best/last metric files.

## Conditioning fixes and generalization pilot, 2026-10-21

All six original services and their 16 experiment cases completed without
logged failures. Direct seed optimization recovered all eight training chains
(mean TM-score 0.9868, RMSD 0.3860 Angstrom). The shared four-chain model also
fit well (0.9824 / 0.4047), whereas the eight-chain normal-contact model reached
0.8925 / 1.4220; seven of eight met the global-shape targets. The local-only
eight-chain model reached 0.9841 / 0.4580. These are training-selected
checkpoints and unequal walltime-limited update counts, not a capacity ranking.

On the 16 development structures, the lifted input had mean TM-score 0.2026
and RMSD 15.7365 Angstrom; the eight-chain normal-contact model gave 0.1893 /
15.8849, and local-only gave 0.2071 / 15.2510. Memorization did not establish
global-fold transfer. The small pairwise information probes did not outperform
sequence separation on held-out distance MAE and predicted no positive
contacts; this does not establish missing information in the representation.

### Normalization diagnosis

The input BatchNorm uses per-structure residue statistics during training,
but running statistics pooled across structures during evaluation. The original
eight-chain checkpoint's train/eval coordinate RMS difference was 0.5887
Angstrom, with training-mode mean TM-score 0.9484 versus evaluation 0.8925.
Gradient cosines did not show pervasive cross-chain opposition.

Preserve the historical `input_normalization: batch` default. Add explicit
`per_structure` normalization as a causal control: use current valid-residue
statistics from each graph in both modes, retaining the same affine weights
and training computation, without tracking running statistics. Batched graphs
normalize separately. Singleton graphs produce the affine bias rather than an
undefined variance. This removes mode mismatch but still removes cross-protein
channel scale differences; it is not automatically the best representation.

Also compare `input_normalization: layer`: per-residue LayerNorm across
feature channels, without moving statistics or cross-residue normalization.
This is a distinct architecture choice, not a checkpoint-compatible repair.
Keep both causal and LayerNorm controls rather than silently changing old
checkpoints. Warm-start guards reject changed normalization/contact settings;
strict resume provenance continues to reject changed source/configuration.
Any affine-weight transfer used to diagnose the historical checkpoint is
explicitly evaluation-only, with discarded BN running buffers recorded.

### Contact controls

Use one shared contact-policy helper for diagnostics and Lightning training:

- `normal`: original inferred contacts plus seed-distance contacts.
- `local-only`: adjacent-row edges only.
- `inferred-only`: cached inferred contacts (including original local edges),
  without seed-distance contacts.
- `seed-only`: adjacent rows plus seed-distance contacts, without inferred
  nonlocal edges.

Policies use only frozen inputs; targets never construct model edges.
Training-only contact audits measure long-range edge quality without changing
thresholds or selecting edges using targets. Do not promote local-only solely
because it improved memorization.

### Matched development experiment

Use a verified subset of 256 training and 64 development-validation chains,
64--128 stored residues and at least 90% valid residues. Preserve calibration
from the source bundle, split separation, selected/excluded IDs and payload
hashes. These are previously monitored refiner splits, not a fresh
cluster-separated test or production-pretraining independence claim.

Run eight fresh-initialization configurations:

| Normalization | Contacts |
|---|---|
| BatchNorm | normal, local-only |
| Per-structure current-stat normalization | normal, local-only, inferred-only, seed-only |
| LayerNorm | normal, local-only |

All share the same structures, seed 42, width 64/depth 4/four heads,
parameter-free tensor-MLP input normalization, float32 forward/backward/AdamW,
float64 gradient accumulation, global batch 16, LR 5e-4, clipping 1, combined
coordinate loss and no scheduler. Each is fixed at 16 epochs / 256 updates /
4,096 training visits, with a 45-minute process ceiling. Incomplete trials
stop the queue and are not accepted as matched comparisons.

Select each checkpoint by the unchanged development evaluation FAPE and retain
every epoch's paired raw/lifted-input/refined metrics. Assess global TM-score,
aligned RMSD, lDDT and backbone geometry together. Require improved FAPE with
no >5% error or >0.01 TM-score/lDDT regression before expansion. Relative
improvements at TM-score near 0.2 remain far from reliable fold reconstruction.
No automatic model promotion, larger-data expansion or retry is authorized by
this pilot's code.

The [generalization runner](../scripts/se3/run_se3_generalization.py) stages fresh
configs and a content-verified frozen subset, snapshots source dependencies,
checks hashes before each child, verifies complete train/validation counts and
finite model/optimizer states, and preserves FAPE-selected checkpoint hashes.
Run sequentially on GPU 1; GPU 0 has an unrelated workload and is left alone.
No Clariden allocation is needed for this bounded pilot.

### Verification before launch

The targeted conditioning/training/loss/refiner/validation suite passed
93 tests, with the pre-existing six-layer numerical failure still expected.
The six initial BatchNorm/per-structure variants passed real-cache CUDA
training, validation, finite-model/optimizer and complete-checkpoint checks.
The two added LayerNorm variants passed the same real-cache CUDA checks.

On the original trained eight-chain checkpoint, per-structure normalization
reproduced the original training-mode coordinates and angles exactly on every
chain. Its train/eval coordinate differences were exactly zero; the original
model's largest component-wise train/eval discrepancy was 5.5312 Angstrom
(distinct from the previously reported mean coordinate RMS). Affine-weight
transfer was diagnostic-only, with no optimizer step or migrated checkpoint.
CUDA backward gradients remained finite. Evidence is
`runs/se3_generalization_smoke_20261021/normalization_gate.json`.

A real-cache single-GPU interrupted/resumed per-structure/local-only run
matched its uninterrupted control **bitwise** in all model and optimizer
tensors after two updates, with identical consumed stream. Evidence is
`runs/se3_generalization_smoke_v2_20261021/resume_equivalence.json`.
This is a local resume check, not a new distributed-topology gate.

### Pilot launch

The eight-case frozen subset/configuration archive is staged at
`runs/se3_generalization_20261021`. Persistent service
`foldtree2-se3-generalization-20261021.service` runs it sequentially on GPU 1.
Each child has a 45-minute ceiling, and the whole service has a six-hour
ten-minute ceiling. `queue.log` records stage transitions; each variant has
its own log, per-epoch paired validation metrics, provenance and atomic
best/last checkpoints. `completion.json` is published only after all expected
epochs/updates, exact validation IDs, finite final model/optimizer state and
FAPE-selected best-checkpoint epoch are verified.
No production weights were changed; the unrelated GPU 0 process was left
running. Queue acceptance/startup is not evidence of improved generalization.

### Completed normalization/contact pilot

All eight cases completed 16 epochs and 256 optimizer updates, with finite
final model/optimizer state and verified validation membership. At the
unchanged development-FAPE-selected checkpoints:

| Case | TM-score | CA RMSD (Angstrom) | lDDT | CA bond MAE (Angstrom) |
|---|---:|---:|---:|---:|
| BatchNorm / normal | 0.22647 | 14.37439 | 0.27472 | 1.25066 |
| Per-structure / normal | 0.22435 | 14.37746 | 0.27408 | 1.25193 |
| BatchNorm / local-only | 0.22802 | 14.32680 | 0.28908 | 0.89462 |
| Per-structure / local-only | 0.22709 | 14.34846 | 0.28913 | 0.89549 |
| Per-structure / inferred-only | 0.22075 | 14.54962 | 0.27822 | 1.21077 |
| Per-structure / seed-only | 0.23079 | 14.44756 | 0.27825 | 1.22676 |
| LayerNorm / normal | 0.22696 | 14.21368 | 0.27703 | 1.21649 |
| LayerNorm / local-only | 0.22717 | 14.45742 | 0.28946 | 0.79642 |

The lifted-input baseline is TM-score 0.20786, RMSD 15.28204 Angstrom and
lDDT 0.20021. Normalization fixes did not solve global folding; local-only
improves local geometry, with no demonstrated global-fold advantage over the
other policies. These are single-seed development results, not statistically
established rankings.

## Frozen contact/AA feature fusion, 2026-10-21

The production geometry branch's cached normalized 30-channel `contact_z`
previously selected binary top-eight contacts, but did not enter residue
features. Its learned contact strength was discarded. The production
`sequence_transformer` AA branch was not called by the old extraction path.

### Input-only enrichment

[enrich_se3_decoder_cache.py](../scripts/se3/enrich_se3_decoder_cache.py) creates
a new, hash-verified bundle without rewriting the parent cache or production
checkpoints. It verifies both frozen checkpoint hashes and the original
graph-dataset hash before extraction. The AA model receives only cached
continuous latent vectors and the original positional encodings: no target
sequence, coordinates, structural graph edges or masks derived from AA labels.
Targets are used separately for the AA quality audit.

The final enriched bundle is `runs/se3_decoder_enrichment_v3_20261021`
(identity
`8d01cbbd3fa52a4261ed8616d25c871661f11beab3fd548befa4dc868bbc6f06`).
Keep its manifest as the authoritative identity; failed and earlier
preparation directories remain diagnostic evidence, not experiment inputs.
Each parent field, split ID and residue mask is preserved; the only appended
entry fields are:

- `aa_probs`: full frozen 20-way softmax distributions, not argmax labels.
  The serialized legacy DNN AA branch returns `[N,1,20]`; extraction removes
  precisely that singleton batch axis and rejects other shapes/nonfinite
  outputs.
- `contact_probs`: symmetric dense probabilities from
  `sigmoid(2.4314587116241455 * contact_z @ contact_z.T - 1.3124585151672363)`,
  with the production probability clamp. Reject unnormalized decoder
  configurations rather than silently applying the normalized-cache formula.

The frozen AA branch has residue-weighted training accuracy **89.289%** and
development accuracy **89.422%**, not exact sequence reconstruction.
`aa_quality.json` retains per-chain accuracy, valid counts and entropy.
Extraction verifies repeated predictions and unchanged frozen weights/no
parameter gradients. Failure records an error and removes any published
success manifest.

### Matched five-case experiment

The isolated
[DecoderFusionRefiner](../foldtree2/src/decoder_feature_fusion.py) wraps the
unchanged SE(3) core with gated adapters back to the original feature width.
Contact and AA adapters have separate LayerNorm/MLP projections and scalar
gates initialized to `tanh(0.1)`. Common branches have identical initialization
across variants; adapter construction preserves the core RNG stream.

| Mode | Additional residue conditioning |
|---|---|
| `baseline` | None |
| `contact` | Frozen contact-decoder node embeddings |
| `aa` | Frozen full AA distributions |
| `both` | Both node-feature branches |
| `both_soft_pairs` | Both, plus confidence-weighted nonlocal contact messages |

The last case is a separate **node-message aggregation** experiment, not a new
edge-aware attention architecture. It aggregates normalized contact embeddings
with frozen contact probabilities, excluding row separation below three and
masked residues. Divide by the number of eligible neighbors, not the sum of
probabilities, so low confidence does not become a strong normalized message.
Do not renormalize the aggregated context. Bias-free projections ensure zero
contact evidence contributes zero pair message.

All five cases use the same 256/64 chains, LayerNorm/local-only core,
width 64/depth 4/four heads, seed 42, original calibrated/lifted seed, optimizer,
loss and deterministic structure order. Lifting happens before fusion using
the original features. Train from scratch for 16 epochs/256 updates, with
global batch 16, float32 model/AdamW and float64 gradient accumulation.
Each variant has a different intentional adapter parameter count, recorded in
provenance. This tests usability of frozen continuous-latent decoder features,
**not** strictly discrete-only reconstruction or encoder/decoder finetuning.

[train_se3_decoder_fusion.py](../foldtree2/train_se3_decoder_fusion.py) retains
input/initial/per-epoch/best/final validation metrics, gates, gradient norms,
source hashes, exact split IDs and best/last model/optimizer snapshots. Select
the best checkpoint using unchanged evaluation FAPE; also compare final
fixed-budget results and paired per-chain TM-score/RMSD/lDDT/bond errors.
Snapshots are explicitly nonresumable in this isolated pilot.

[run_se3_decoder_fusion.py](../scripts/se3/run_se3_decoder_fusion.py) stages
source/configuration snapshots, verifies bundle/source/configuration identity
before every case, enforces a 45-minute child ceiling and stops on failure.
The persistent queue is `runs/se3_decoder_fusion_v2_20261021`, service
`foldtree2-se3-decoder-fusion-20261021.service`, sequentially on local GPU 0.
The resource ceiling is 3 hours 45 minutes of child GPU time and four hours
of service walltime. No cluster allocation or automatic expansion.
The service is active and the baseline case has reached eight optimizer
updates with finite gradient norm 9.41558, using approximately 2.5 GiB on GPU 0.
The earlier normalization/contact queue has completed. Fusion outcomes are
pending; `queue.log` and each mode's `events.jsonl` retain live progress.

### Verification and interpretation gates

The targeted fusion/conditioning/training/loss/refiner/validation suite passed
**110 tests**, with the pre-existing depth-six failure still expected.
Real-cache CUDA smoke runs completed all five modes for two updates each,
including full validation and finite optimizer/model checks. Final smoke
evidence is `runs/se3_fusion_smoke_v3_20261021`.
Its CUDA branch audit also verifies nonzero gradients for every active adapter,
measurable sensitivity to each enabled input, exact checkpoint reload and
unchanged inference when only supervision targets change. Its preservation
audit checks all 320 enriched entries against every original parent field.
The adapter tests cover masks/nonfinite rejection, soft-contact confidence,
shared initialization, rigid equivariance, train/eval equality, gradients,
checkpoint reload and target-independent inference. Enrichment tests check
parent preservation, split identity, AA-only input graph and failure cleanup.

Do not infer improved generalization from the smoke checks or queue startup.
The development holdout has already been monitored and is not certified
independent of production pretraining. Promote a convincing paired global-fold
gain to multi-seed confirmation and a fresh family/cluster-separated test,
not another round of holdout-driven hyperparameter tuning.

## Long-range restraint sufficiency diagnostic, 2026-10-21

Before adding a learned distance head, test direct coordinate fitting under
different restraint sources. This adds no network and does not modify the
active decoder-fusion experiment.
[diagnose_se3_restraints.py](../scripts/se3/diagnose_se3_restraints.py) reuses the
exact eight training chains and calibrated/lifted initial coordinates of the
original `coordinates_seed` control. It verifies the original bundle/panel
identity and frozen decoder checkpoint hashes, then stages only those entries.
No production forward calls occur.

Every case has the same generic local priors: SmoothL1 CA-bond length error
against 3.8 Angstrom, plus nonadjacent excluded-volume penalty below
3.6 Angstrom. There is **no native local-distance, angle, torsion, FAPE or
alignment loss**. Exact native local torsions would themselves supply most
of the answer and confound the long-range-information test.

All long-range pairs are unordered, valid-residue pairs with stored-row
separation at least 12. Cases are:

| Case | Geometric restraint |
|---|---|
| `local_only` | None beyond shared generic local priors |
| `predicted_contacts` | Frozen calibrated contact probabilities on all eligible pairs |
| `predicted_top_pairs` | Decoder-ranked top 2L pairs, contact upper bounds only |
| `oracle_contacts` | Native binary CA contact/noncontact inequalities on all eligible pairs |
| `oracle_distances_uniform` | Native CA distances for 2L uniformly sampled eligible pairs |
| `oracle_distances_decoder_pairs` | Native CA distances for the same decoder-ranked 2L pairs |
| `oracle_distances_all` | Native CA distances for every eligible long-range pair |

Here L is valid residue count, and budgets are capped at the eligible-pair
count. Uniform sampling has fixed seed 42; rank selection and pair budgets
do not depend on native geometry. The rank-only predicted control separates
contact ranking from probability-calibration effects.

Contact-map losses combine SmoothL1 upper-bound error above 8 Angstrom and
lower-bound error below 8 Angstrom, scaled by 4 Angstrom, weighted by p and
1-p. A fixed positive multiplier of 10 is shared by predicted and oracle
contact maps, not estimated from native contact prevalence. Unlike a logistic
distance classifier, native inequalities can have zero restraint error at
native coordinates without demanding arbitrarily extreme separation.
Sparse/native distance losses use SmoothL1(distance/4). Different restraint
types have different gradient scales; this is a sufficiency diagnostic,
not an optimized architecture/loss ranking.

**Contact-definition caveat:** the graph builder constructs decoder training
contacts from CB coordinates (with its missing-CB handling), whereas this
experiment fits CA coordinates. Applying decoder probabilities at an
8-Angstrom CA threshold is an approximation, not a certified calibration
test against the original training labels. `contact_audit.json` records
per-chain CA prevalence, decoder average precision, Brier error and top-2L
precision. One original panel member has zero native long-range CA contacts;
do not remove it after looking at results or interpret its contact-only
failure as a capacity limit.

All cases use AdamW LR 0.05, no weight decay, averaged eight-chain objective,
gradient clipping 1, 2,000 updates or 20 minutes per case. Select best
coordinates using the case's own restraint objective, never native evaluation
metrics; also preserve final fixed-budget coordinates/metrics and step count.
No optimizer/source/bundle tuning or automatic restart between cases.

Distance restraints alone cannot distinguish mirror images. Report the
existing proper-rotation CA RMSD/TM-score and a separately labeled
reflection-allowed RMSD; do not count a reflected fit as successful folding.
Native contacts and distances are explicitly privileged inputs here, not
normal refiner inputs, learned generalization or evidence of discrete-only
reconstruction.

Validation passed 41 restraint/diagnostic/fusion tests. Real-cache CUDA smoke
checks completed all seven modes on the original eight-chain panel for ten
steps each; evidence is `runs/se3_restraint_smoke_v2_20261021`.
The full comparison is staged at `runs/se3_restraints_20261021`, service
`foldtree2-se3-restraints-20261021.service`, on spare local GPU 1. Its ceiling
is 140 minutes child GPU time and 150 minutes service walltime; no Clariden
allocation. Existing GPU 0 workloads are untouched.
Startup was verified: local-only reached 500 updates with finite objective
0.002884 and gradient norm 0.000123, using approximately 490 MiB on GPU 1.
This checks execution only, not reconstruction success; all seven outcomes
remain pending.

If sparse native restraints rescue folds while predicted restraints do not,
prioritize learned restraint quality. If contacts fail but sparse distances
work, prioritize distance distributions over binary contacts. A failure even
with dense native distances requires checking coordinate optimization,
chirality and missing local constraints before drawing information-content
conclusions. Only after a direct-optimization rescue should a matched neural
refiner with the same oracle restraints test the network bottleneck; that
follow-up is not automatically launched by this runner.

## Frozen-feature distance-distribution head, 2026-10-21

[PairwiseDistanceHead](../foldtree2/src/distance_restraints.py) is a new opt-in
module, independent of the production decoder and active SE(3) fusion runs.
The encoder, contact decoder and AA decoder stay frozen; inputs are read from
the verified enriched cache. This is continuous-latent-assisted geometry
prediction, not a discrete-only claim.

The head normalizes the original 40-channel residue features, 30-channel
contact embeddings and 20 AA probabilities separately, projects them to
64-channel node features, and forms symmetric pair sums, absolute differences
and products. The pair MLP additionally receives log row separation and the
frozen contact-embedding cosine. Pair ordering does not change logits.
Inference never reads native coordinates, seed coordinates, true AA labels
or structural edges; frozen input tensors are detached from autograd.
Masked residues cannot enter pairs. Pair chunks limit intermediate memory;
the final returned tensor still has O(number-of-pairs) storage.

### Distance and restraint semantics

Output is a ten-way distribution with intervals
`[0,4), [4,6), [6,8), [8,10), [10,12), [12,16), [16,20), [20,24), [24,32), [32,infinity)`
in Angstrom. Exact boundary values enter the higher bin. Summing the first
three bins gives CA-contact probability below 8 Angstrom.

The last bin is an open-ended tail, **not** an assertion that distance equals
32 Angstrom. `capped_distance_mae_angstrom` uses finite-bin midpoints and
represents the overflow bin by 32, with native distances capped at 32 as well;
this is a coarse diagnostic, not uncensored distance MAE.

The accompanying `distribution_restraint_loss` consumes coordinate pairs and
frozen predicted distributions. It penalizes the negative log of a mixture
of soft interval-compatibility weights, with default 1-Angstrom transitions.
The first/last intervals have only an upper/lower transition respectively.
Thus the overflow bin imposes no spurious upper distance bound. It preserves
mixture uncertainty rather than forcing every pair to an expected distance.
It is a soft restraint score, not a calibrated continuous distance density.
Distances and these restraints cannot resolve chirality on their own.

### Matched training/evaluation path

[train_distance_restraints.py](../foldtree2/train_distance_restraints.py)
compares two fresh-initialization, identical-parameter-count heads:

- `sequence`: all residue features and contact cosine zeroed; retains row
  separation. This controls for the geometry learnable from sequence spacing.
- `decoder`: all frozen residue/contact/AA features enabled.

Use the same enriched 256/64 training/development splits. Sample up to 512
unordered valid pairs per chain per update, uniformly without replacement
from stored-row separations at least 12. One optimizer update per training
chain; shuffle order and pair sampling deterministically using seed 42.
Each mode resets model/sampling seeds and gets the same sampled pairs.
Unweighted cross entropy preserves the distance-bin distribution, instead
of disguising contact imbalance with a class-weighted probability target.
Compare against a smoothed empirical bin prior fitted on training pairs only.

The bounded pilot uses 16 epochs (4,096 updates per mode), hidden width 64,
AdamW LR 1e-3/weight decay 0.01, clipping 1, float32 and two CPU threads.
Each mode has a 15-minute optimization ceiling; time-limited partial epochs
are explicitly recorded and must not be treated as equal-budget completions.
No validation-selected checkpoint or hyperparameter sweep: retain final
fixed-budget weights and per-epoch metrics.

Evaluate **all** eligible nonlocal pairs in each validation chain, including
contact-negative chains, not a balanced or target-selected pair subset.
Report pair/macro-chain NLL, training-prior NLL, bin accuracy, capped MAE,
CA-contact Brier score, precision/recall, tie-aware average precision, top-L
counts and per-chain metrics. Global top-L metrics rank across all chains
with total valid-length budget; per-chain top-L results are also retained.
The frozen contact decoder's separate CA-based audit remains subject to its
CB-training-label mismatch. Undefined precision/recall/AP are explicit nulls,
with true/predicted contact counts.

Each mode saves source/bundle/split provenance, sampled training objectives,
final full training/validation reports, an exact-reload-checked model/optimizer
checkpoint, and `validation_restraints.pt`. That export contains only IDs,
masks, pairs, predicted probabilities, bin edges and checkpoint/bundle hashes;
no native coordinates or labels. Snapshots are explicitly nonresumable.
This provides an opt-in handoff to a coordinate optimizer; it does **not**
silently change the production SE(3) model or launch learned-restraint
coordinate reconstruction.

### Pairwise error audit after completed reconstruction

[audit_distance_pair_errors.py](../scripts/se3/audit_distance_pair_errors.py)
decomposes all 213,574 eligible pairs on 64 development chains. Evidence:
`runs/se3_pair_error_audit_v2_20261021/audit.json`. It checks export/checkpoint/
bundle identity and reconstruction panel hashes, evaluates conditional distance
bias, calibration, sequence-separation strata, per-chain errors, and the actual
restraint score at native and fitted coordinates. Eighteen focused
audit/head/reconstruction tests passed; the audit performs no model updates.

The full-feature head improves NLL on 52/64 chains and capped MAE on 54/64
versus sequence-only. Train/validation NLL are 1.754/1.766, and capped MAE
5.970/5.998 Angstrom. This does not look like a large training/validation gap;
it does not prove sufficient capacity or convergence.

However, predictions remain weakly pair-specific. For native distances
4--6 Angstrom, mean capped predicted distance is 19.01; for 6--8 it is 19.70;
for 8--10 it is 20.04. For native distances at least 32, mean capped prediction
is 23.89. These expectations are diagnostics, not the coordinates directly
requested by the mixture restraint. Mean maximum-bin probability is only
0.271, and entropy is 1.807 nats. Broad distributions are not by themselves
proof of miscalibration, but they provide weak pair discrimination.

Only 3,472/213,574 pairs (1.626%) are native CA contacts. Among 2,156 pairs
with predicted contact probability 0.1--0.2, mean prediction is 0.1144 and
observed contact frequency 0.0993. There are no probabilities at least 0.2.
Thus lowering a 0.5 threshold would not reveal a hidden set of highly reliable
contacts. Per-chain top-L selection pooled across chains gives precision
0.06471, versus sequence-only 0.02537. This differs from the earlier
globally ranked total-L result (0.08387): do not conflate ranking protocols.

The distance NLL improvement is concentrated at larger row separations:
sequence/full-feature NLL is 1.8846/1.8796 for separations 12--23,
1.8689/1.8283 for 24--47, and 1.7331/1.6098 for at least 48.

The decisive solver check is unfavorable: **on all eight reconstruction
chains, fitted incorrect coordinates have a lower predicted-restraint score
than native coordinates**. Mean native/fitted restraint scores are
1.8094/1.4196 for full features. Mean nonlocal distance expands from native
23.65 and seed 24.06 to fitted 29.46 Angstrom; nonlocal distance MAE is
11.06 at the seed versus 11.37 after fitting. The optimizer is succeeding
at its stated objective, but that objective does not favor the correct fold.

For every true CA-contact pair in this validation set, the individual
full-feature soft-mixture restraint derivative at its native distance points
toward increasing that distance. This is a scalar pair-restraint diagnostic,
not a claim about the net Cartesian force including other pairs/local priors.
An independent 2--64.5-Angstrom grid search puts the mean preferred distance
of native 4--6-Angstrom pairs at 19.92 and native 6--8 pairs at 22.29.

There are two distinct problems to resolve before scaling: weak near-contact
discrimination in the head, and conversion of broad unequal-width/open-tail
bin probabilities into geometric energies. The current soft-interval
compatibility score is not a normalized continuous density; wide/tail bins
can contribute more compatibility than narrow bins. The audit establishes
the wrong geometric preference, but does not quantify how much is caused by
this score versus the learned distributions. Do not conclude that a density
correction alone will repair folding.

Next gates: check small-panel distance-head memorization and stratified
per-bin learning first; compare native-versus-decoy energy with a
training-fitted background/reference correction before further coordinate
runs. Any balanced sampling/class-weight change must preserve or explicitly
correct probability calibration. Freeze development choices before a fresh
test; do not select contact thresholds on the monitored validation set.

## Distance learning and energy acceptance gates, 2026-10-21

[diagnose_distance_learning.py](../scripts/se3/diagnose_distance_learning.py) runs
the next bounded diagnostics without modifying existing heads, caches or
historical coordinate scores.

### Memorization before generalization

Select a deterministic length-spread panel of eight **training** chains from
the enriched bundle, requiring at least one true long-range CA contact per
chain so contact learning is measurable. This is explicitly target-conditioned
training-panel selection, not held-out evidence; report the number of excluded
contact-negative chains. The one-chain case is the first member of this same
panel. Both fresh-initialization heads use the existing width-64 pairwise
architecture and seed 42. Do not add native/seed coordinates, target sequences
or residue identifiers as inference features.

Optimize the mean per-chain **all-eligible-pair** cross entropy with AdamW
LR 1e-3, no weight decay, clipping 1, at most 4,000 updates or ten minutes per
case. Both cases use exactly the same eligibility (valid residues, row
separation >=12), not a target-balanced subset. Retain per-bin NLL, bin
accuracy and contact AP/precision/recall every 100 updates. Select best weights
by training-panel NLL, check exact reload, and report final as well as best
metrics. Snapshots are explicitly nonresumable.

Predeclared success requires NLL <=0.25, bin accuracy >=0.90 and contact
AP, precision and recall all >=0.80. Undefined precision cannot pass.
Stop early only when that complete gate passes; low average error with zero
contact recall is not success. Evaluate the memorized head's reference score
on native, seed and fixed 4-Angstrom native-noise coordinates separately.
These are training energy checks, not production refiner inputs.

### Opt-in normalized/reference-relative energy

[distance_energy.py](../foldtree2/src/distance_energy.py) adds a separate
score rather than silently changing `distribution_restraint_loss`.
Its smooth bin memberships are logistic-CDF differences with open first/last
tails and sum to one at every distance, even with unequal widths. A stable
log-domain formulation avoids subtraction of near-equal sigmoid values.

The reference is a Laplace-smoothed distance-bin population fitted only on
the head's exact training split, separately for row separations 12--23,
24--47 and >=48. Evaluate the per-pair score
`-log(sum_k membership_k(distance) * predicted_probability_k / reference_k)`.
If predictions equal the reference, energy and its distance derivative are
zero: a background-only prediction must not create folding forces.
This is a background-relative statistical potential, not an absolute
continuous density; it can have negative values. Reference correction does
not assert that a weak predictor becomes useful.

Compare three fixed scoring controls: historical mixture, unit-partition
mixture without reference, and unit-partition/reference score. Do not
optimize coordinates during this gate. Score all 64 monitored development
chains against native coordinates, original seed, fixed-seed Gaussian native
noise (1/4/8 Angstrom per Cartesian axis), and coordinate-permutation decoys.
On the prior eight-chain reconstruction panel, include all three already
fitted decoy sets. Native masks and coordinate IDs must match verified
prediction/checkpoint/bundle identities. Permuted/noised chains can violate
local geometry; all scores here are the pair potential alone.

Predeclared acceptance for the full-feature/reference score requires native
to win on >=75% of chains **separately** against seed, 4-Angstrom noise,
8-Angstrom noise and the previous full-feature fitted decoys. Report all
other decoys too, rather than selecting whichever set supports the method.
No acceptance threshold is tuned after inspecting development results.

### Current energy result and sampling guard

Real-cache CUDA smoke execution already completed the full energy audit.
The corrected score **fails** acceptance: native wins against seed on only
7/64 chains and against the previous full-feature fitted decoys on 3/8.
It wins against 4-/8-Angstrom noise on 49/64 and 52/64 respectively. Merely
normalizing memberships without reference still prefers the wrong fitted
decoys on all eight chains. This is an unsuccessful candidate correction,
not a repaired folding objective; do not launch coordinate optimization
with it or loosen the gate to make it pass.

Contact-focused sampling is implemented only as a **gated small training-panel
control**, not a held-out model improvement claim. Sample equal numbers of
contact/noncontact pairs with replacement and weight each per-pair CE by
population-stratum-probability / sampled-stratum-probability. This estimates
the original population CE rather than relabeling a balanced sample as
calibrated probabilities. Single-class chains use explicit uniform sampling.
Compare to a fresh matched uniform-sampling model (512 draws/chain/update,
1,000 updates or ten minutes per case); record full-panel metrics.

The sampling pilot can run only if **both** memorization cases and the energy
gate pass. Otherwise publish `sampling_pilot.json` with blocked status and
individual gate results. The failed reference-energy gate therefore blocks
this expansion with the currently trained heads.

The runner writes source snapshots, cache/input hashes, exact panel IDs,
training-only reference populations, per-chain native/decoy scores,
memorization curves/checkpoints and explicit completion/failure records.
Real-cache CUDA smoke evidence is
`runs/se3_distance_learning_smoke_v2_20261021`.
The bounded full run is `runs/se3_distance_learning_20261021`, persistent
service `foldtree2-se3-distance-learning-20261021.service` on idle GPU 1.
Ceiling: 20 minutes for the two memorization cases, an additional 20 minutes
only if both diagnostic gates permit the sampling control, and 45 minutes
service walltime. No cluster allocation or production checkpoint updates.

Launch was verified on GPU 1 (approximately 424 MiB). The one-chain head
passed its complete gate after **800 updates**: NLL 0.13396, bin accuracy
0.93224, contact AP 0.95855, precision 0.80 and recall 0.94118
(16/17 native contacts retrieved, 20 predicted). On this memorized chain,
reference energies are native -1.4061, seed 14.7437 and fixed 4-Angstrom
native-noise decoy 0.1904. Thus the head can learn close-pair constraints and
the reference score can recognize the training native when predictions are
strong; this does not repair the existing generalizing head. The eight-chain
case is running. The focused energy/memorization/sampling and related
distance/fusion suite passed **56 tests**.

### Longer fixed-architecture memorization control

The initial eight-chain run reached its ten-minute limit after 3,599 updates:
best NLL 0.87857, bin accuracy 0.65886, contact AP 0.61813, precision 0.85976
and recall 0.35969. It did not pass the learning gate. Its reference energy
nevertheless preferred native coordinates over seed and fixed 4-Angstrom
noise on all eight training chains. Neither result establishes held-out
fold reconstruction.

The user-approved longer comparison uses the exact same width-64 head,
panel/seed, all-pair objective, optimizer and acceptance gates, extending only
the budget to **20,000 updates or 3,600 seconds per case**. It starts fresh:
previous best checkpoints are explicitly nonresumable and lack optimizer
state. The one-chain control is rerun with the same early success gate;
the primary comparison is the eight-chain curve at matched update counts.
Do not describe this as continuing the earlier optimizer trajectory.

Outputs: `runs/se3_distance_learning_long_20261021`; persistent service
`foldtree2-se3-distance-learning-long-20261021.service` on GPU 1.
Service walltime ceiling is 130 minutes, covering two possible one-hour
memorization cases and evaluation. The existing held-out energy gate still
fails and blocks the sampling pilot; no wider model, coordinate run,
cluster allocation or automatic retry is authorized by this experiment.

### Fixed-panel width comparison, 2026-10-22

The longer width-64 control completed 20,000 updates in 3,402 seconds.
Its best training-NLL checkpoint (update 19,900) reached NLL 0.51616,
bin accuracy 0.80616, contact precision 0.88410, recall 0.83673 and
AP 0.93607. Contact criteria pass, but the complete distance-learning gate
does not. Continued learning is not evidence by itself of insufficient
capacity; compare capacity and optimization budget explicitly.

[compare_distance_capacity.py](../scripts/se3/compare_distance_capacity.py)
runs widths **64, 128 and 256** sequentially on GPU 1, with respectively
27,454 / 95,678 / 355,006 parameters. Every case starts fresh from seed 42
on the exact previously selected eight training IDs, with unchanged feature
inputs, symmetric pair construction, all-eligible-pair equal-chain CE,
AdamW LR 1e-3/no weight decay and clipping 1. Only width changes.
The width-64 case extends the optimization control to **40,000 updates**;
each width has a **two-hour** training ceiling. No checkpoint resume,
validation selection or production-model changes.

Retain metrics every 100 updates and at the first observation after each
600-second walltime landmark, including actual elapsed time and update count.
Compare shared update counts separately from shared walltime, using observed
timestamps rather than assuming identical throughput. Different widths may
complete unequal updates at the time ceiling. Log the first full acceptance
gate crossing, but do not early-stop: continued fixed-budget results remain
available across widths. Select best weights by training NLL and retain
final model/optimizer state, per-bin/contact reports and exact-reload checks.
Snapshots remain explicitly nonresumable.

Real-cache GPU smoke runs completed all three widths with finite gradients
and exact reload: `runs/se3_distance_capacity_smoke_20261022`.
Twenty-two targeted capacity/learning/distance tests passed.
The persistent comparison is `runs/se3_distance_capacity_20261022`, service
`foldtree2-se3-distance-capacity-20261022.service` on GPU 1. Ceiling is six
GPU-hours of training plus 15 minutes service overhead; no cluster allocation
or automatic expansion. GPU 0's unrelated workload is untouched.
Success here establishes only training-panel capacity, not held-out geometry.

### Redirecting width comparison to development generalization

At user request, do not spend the remaining queue on width256 memorization.
[queue_distance_generalization.py](../scripts/se3/queue_distance_generalization.py)
waits at most 30 minutes for width128 to complete its 40,000 updates, then
stops the capacity service, preserving its width64/128 checkpoints and logs.
If width256 starts during the transition, stop it and record that explicitly;
do not claim it completed or failed a learning gate.

Queue fresh decoder-feature heads at widths64/128/256 on the identical
256-training/64-development enriched bundle. Keep seed42, uniformly sampled
512 eligible pairs per chain, AdamW1e-3/weight decay0.01, clipping1 and all
original trainer semantics unchanged. Extend to 128 epochs/32,768 updates
per width or one hour optimization per width. Evaluate all nonlocal pairs
every epoch; retain final fixed-budget training/validation reports, per-chain
contact/bin metrics, exports and exact-reload checkpoints. Compare training
convergence and matched update/walltime curves; a time-limited case must be
labeled rather than treated as an equal-budget completion. No held-out-driven
checkpoint selection or coordinate reconstruction.

The prior 16-epoch width64 sequence-only run remains a historical baseline,
not a new matched 128-epoch sequence-only control. This comparison isolates
head width, not all architectural explanations. Existing held-out energy
failure remains a reconstruction gate.

Queue outputs: `runs/se3_distance_generalization_widths_20261022`, service
`foldtree2-se3-distance-generalization-20261022.service` on GPU1.
Ceiling: 30-minute parent wait plus three one-hour optimization cases with
evaluation/export overhead; service cap four hours. Stop on errors, no retry
or cluster submission. Fourteen queue/trainer tests passed.

Example bounded comparison:

```bash
python -m foldtree2.train_distance_restraints \
  --cache runs/se3_decoder_enrichment_v3_20261021 \
  --output runs/se3_distance_head_20261021 \
  --device cuda:1 --epochs 16 --seconds 900
```

CPU and CUDA smoke runs on 16 training/four validation chains completed two
epochs per mode, with finite gradients, target-independent inference,
normalized export probabilities and exact checkpoint reload. CUDA evidence:
`runs/se3_distance_head_cuda_smoke_20261021`. These smoke results establish
execution, not better held-out distance prediction or fold reconstruction.

The persistent comparison is
`foldtree2-se3-distance-head-20261021.service` on local GPU 1, now free after
the restraint diagnostic. Its walltime ceiling is 35 minutes, with 30 minutes
total optimization allowance and extra evaluation/export time. GPU 0's
fusion run and unrelated workloads remain untouched; no cluster allocation.
Launch was verified at 960 finite optimizer updates in the sequence-only
control, using approximately 414 MiB on GPU 1. Outputs are
`runs/se3_distance_head_20261021`; the service log is
`runs/se3_distance_head_20261021.queue.log`. Results remain pending.
The targeted distance/restraint/fusion/SE(3) suite passed 136 tests, with one
pre-existing expected depth-six numerical failure.

### Restraint diagnostic motivating this test

All seven earlier direct-coordinate controls completed 2,000 steps. At their
own-objective-selected coordinates, mean TM-score / proper CA RMSD were:

| Restraints | TM-score | RMSD (Angstrom) |
|---|---:|---:|
| Local priors only | 0.2304 | 15.038 |
| Decoder probabilities | 0.1754 | 14.001 |
| Decoder top pairs | 0.2249 | 14.829 |
| Native CA contacts | 0.2922 | 12.195 |
| Native distances, uniform 2L | 0.4465 | 7.512 |
| Native distances, decoder-ranked 2L | 0.2699 | 16.072 |
| All native long-range distances | 0.7897 | 3.356 |

The dense-distance case has reflection-allowed RMSD 0.747 Angstrom and lDDT
0.962, but that does not make mirror images valid folds. Sparse pair coverage
and chirality remain important. These privileged coordinate-fitting results
motivate predicting distances and testing their quality against sequence-only
controls; they do not promise that the new head can learn those distances.

## Reconstruction with the trained distance head, 2026-10-21

Both distance-head cases completed 16 epochs/4,096 updates. Full-feature
development pair NLL is 1.76601 versus sequence-only 1.82595 and empirical
training-prior 1.85192. Capped distance MAE is 5.99768 versus 6.52793 Angstrom;
global top-L contact precision is 0.08387 versus 0.02606. Neither case predicts
any contacts at probability 0.5. These are distance-prediction gains on the
monitored development holdout, not demonstrated fold recovery.

[reconstruct_distance_restraints.py](../scripts/se3/reconstruct_distance_restraints.py)
now consumes the frozen validation distributions for direct coordinate
optimization. It verifies checkpoint/export/bundle identities, exact head
validation membership, canonical complete nonlocal pair coverage, normalized
probabilities, identical bin edges and exact residue masks. Selection is a
fixed length-spread panel of eight validation chains, chosen before coordinate
evaluation; no training IDs or target-dependent pair selection are allowed.

Three cases start from exactly the same calibrated/lifted decoder coordinates:
generic local priors only, sequence-only distance head, and full-feature
distance head. All share CA-bond 3.8-Angstrom and nonadjacent 3.6-Angstrom
exclusion priors. Predicted-head cases add the unit-weight soft interval-mixture
restraint score with 1-Angstrom transitions. Native coordinates/angles are
evaluation-only, never objective inputs. There are no oracle contact,
distance, local-angle or torsion terms.

Use AdamW LR 0.05, no weight decay, clipping 1, mean eight-chain objective,
2,000 updates or 20 minutes per case. Select coordinates by their own
optimization objective, not native metrics; retain initial, selected and final
coordinates/metrics plus complete step counts, proper and reflection-allowed
RMSD, TM-score, lDDT and bond MAE. This is a direct coordinate solver using a
frozen learned distance head, **not** joint SE(3) training or a production
default. Compare paired per-chain results, not just the optimization loss
whose scale differs between cases.

The focused distance-head/reconstruction suite passed 16 tests. Real-cache
CUDA smoke checks completed ten updates for all three cases:
`runs/se3_distance_reconstruction_smoke_20261021`. The full comparison is
`runs/se3_distance_reconstruction_20261021`, service
`foldtree2-se3-distance-reconstruction-20261021.service`, on idle GPU 1.
Resource ceiling: 60 minutes total child optimization allowance, 65 minutes
service walltime, no cluster allocation or automatic retries. Native restraint
and head-training runs remain unchanged.

## Representative-count scaling, 2026-10-22

The next test increases training diversity rather than capacity. The completed
256-training/64-development width comparison did not show better held-out
prediction from larger heads. That does not rule out improvement at a larger
data scale, including a possible threshold-like transition.

[scale_distance_data.py](../scripts/se3/scale_distance_data.py) stages nested
256, 1,024 and 2,048 training representatives from the existing 4,096/512 cache.
All cases use the same 256 validation representatives: the original monitored
64 plus 192 additional IDs not previously used to evaluate the distance head.
The original 256 training IDs and original split membership are retained.
Eligibility requires at least 90% valid residues and an eligible nonlocal pair;
additional IDs are selected deterministically, without selecting on target
geometry or prediction quality. The available eligible pool is 2,157 training
and 267 validation representatives.

The dataset originates from AFDB clusters, so representative holdout is the
working approximation to family holdout. No cluster-membership/homology audit
is available here, and frozen production-model pretraining overlap is unknown.
Do not claim certified family or pretraining independence. Expanded cohorts
also include longer chains than the original short-chain panel: count and
length distribution therefore change together.

Use the same width-256 head, frozen inputs, population cross-entropy, AdamW,
512 sampled pairs per chain, seed 42 and 128-epoch limit in all cases. Compare
both equal epochs and equal optimizer updates; larger sets receive more total
updates at equal epochs. Each case has a two-hour elapsed training/evaluation
allowance, checked after updates, with bounded final-evaluation/export overhead.
The single-GPU service ceiling is 6 hours 45 minutes, including extraction and
subprocess overhead, with no automatic retries or cluster allocation.

Full-pair validation is saved after every epoch. Separate macro-chain reports
for the original 64 and additional 192 are retained with actual update counts
and elapsed times, including partial final epochs. Final checkpoints are not
chosen by validation. This tests distance prediction, not successful coordinate
reconstruction; the failed native-versus-decoy restraint-energy gate remains
in force.

The expanded-length CUDA smoke at
`runs/se3_distance_data_scaling_smoke_20261022` completed 32 finite updates on
16 training/four validation chains of lengths 245-256, with frozen AA/contact
enrichment and exact checkpoint reload. Full outputs are designated
`runs/se3_distance_data_scaling_20261022`, service
`foldtree2-se3-distance-data-scaling-20261022.service`, on local GPU 1.
GPU 0's unrelated workload is not modified.

### Completed first scaling comparison

All three cases completed 128 epochs. On the identical 256-chain holdout:

| Training representatives | Updates | Final pair NLL | Capped MAE (Angstrom) | Global top-L contact precision |
|---|---:|---:|---:|---:|
| 256 | 32,768 | 1.7690 | 6.063 | 0.0528 |
| 1,024 | 131,072 | 1.6695 | 5.783 | 0.0760 |
| 2,048 | 262,144 | 1.6593 | 5.700 | 0.0798 |

At 32,768 updates, NLL was respectively 1.7690, 1.6628 and 1.6668:
the diversity benefit is not explained solely by more updates at equal epochs.
Both the original64 and additional192 macro-chain reports improve. This is
evidence of improved held-out distance prediction, not a demonstrated abrupt
transition or reliable fold reconstruction.

### Next expansion: 4,096 and 5,120 representatives

[extend_distance_data.py](../scripts/se3/extend_distance_data.py) selects 3,072
additional training representatives from the same original graph dataset,
retaining all 2,048 preceding training IDs and exactly the same 256 validation
IDs. Additions exclude every ID in the original 4,096/512 cache as well as the
preceding cohort. This prevents any formerly held-out ID entering training.
Selection is seed42, independent of prediction quality, restricted to 64-256
residues, at least 90% valid residues and an eligible nonlocal pair. The larger
sets are nested, but are not certified homology-independent families. All 256
validation IDs are now monitored development data, not an untouched test set.

Frozen production extraction verifies dataset/checkpoint/extractor hashes,
deterministic repeated extraction and unchanged model state/gradients. Raw
retained entries are copied without modification. The existing enrichment and
distance-training paths are reused. New heads start fresh at each count;
width256, population CE, optimizer, seed42, 512 pairs per chain and the 128-epoch
limit remain unchanged. Matched-update comparisons remain necessary.

An initial attempt to select 6,144 additional chains for an 8,192 cohort found
only 3,346 eligible additions in the local dataset after prior-ID exclusions.
Rather than silently relaxing confidence or length criteria, this experiment
therefore uses nested 4,096 and 5,120 cohorts. Further doubling requires a
larger input dataset or an explicitly separate broader-eligibility experiment.

Outputs: `runs/se3_distance_data_scaling_5120_20261022`; the nested training
results will be under `comparison/`. Persistent service:
`foldtree2-se3-distance-data-scaling-5120-20261022.service`, local GPU1 only.
Resource ceilings: eight elapsed training/evaluation hours per case plus at
most 15 minutes subprocess startup/final-export overhead, 20 hours total
service walltime including extraction/enrichment, and 64 GiB host memory for
the larger full-pair final-training audit. No cluster allocation, automatic
retries, production changes or coordinate fitting.

## Trainable residue-context comparison, 2026-10-23

The 4,096 and 5,120 scaling cases completed 128 epochs. Their held-out pair
NLLs were 1.65837 and 1.65638; capped MAEs 5.795 and 5.678 Angstrom; global
top-L contact precisions 0.08159 and 0.08280. These modest gains do not
demonstrate an abrupt generalization transition or reliable fold recovery.

The next controlled test holds the enriched 5,120/256 cache, population CE,
pair sampling, AdamW LR0.001/weight-decay0.01/clip1, seed42 and 128-epoch
limit fixed. All heads start fresh; the existing final width256 result is the
baseline. Exact equality of baseline initialization, configuration, weights
and real-cache logits against its archived implementation was verified.

Two new cases distinguish whole-chain information exchange from parameter
count:

| Case | Hidden width | Context | Parameters | GPU | Training/evaluation allowance |
|---|---:|---|---:|---|---|
| Existing baseline | 256 | None | 355,006 | Completed | Completed 128 epochs |
| Residue Transformer | 256 | 2 layers, 8 heads | 1,936,078 | 1 | 12 hours |
| Parameter-matched MLP | 612 | None | 1,937,782 | 0 | 8 hours |

The Transformer follows the existing node MLP with pre-normalized full
self-attention and four-times-width GELU feed-forward blocks, zero dropout,
residual connections and final LayerNorm. Learned per-head relative attention
bias uses signed residue-row offsets clipped to [-32,32]; exact pairwise log
separation remains available downstream. Invalid residues cannot contribute as
attention keys or supervised pair endpoints. The symmetric pair head is
unchanged. Frozen inputs are detached; native/seed coordinates, target angles,
AA labels and native-contact masks are never context inputs. This is residue
context, not AlphaFold-style pair triangle updates.

The default remains zero context layers, with historical state/config and
checkpoint compatibility. CLI opt-in: `--context-layers 2 --context-heads 8`.
Tests verify exchange from a third residue to a queried pair, masked-residue
isolation, pair symmetry/chunking, target independence, frozen gradients,
learning, and exact checkpoint reload through the bounded trainer.

Output root: `runs/se3_distance_context_20261023`, with cases
`transformer` and `matched_mlp`. Services:
`foldtree2-distance-context-transformer-20261023.service` (13-hour walltime)
and `foldtree2-distance-context-mlp-20261023.service` (9-hour walltime).
Each has a 64-GiB host-memory ceiling for the full-pair final training audit.
Both GPUs were idle before launch; no other jobs are stopped or changed.

Compare equal epochs/optimizer updates first, and report walltime separately.
Use final fixed-budget checkpoints, not validation-selected weights. If a case
hits its time ceiling, report actual updates and compare only shared learning
curve landmarks. Improvements over the baseline alone are insufficient to
attribute gains to communication; compare against the parameter-matched MLP.
This remains a single-seed monitored development experiment. Better distances
must still pass native-versus-decoy energy checks before new folding claims.

## Frozen-contact attention bias, 2026-10-23

The next variant adds `alpha[layer,head] * (contact_z @ contact_z.T)` to
attention logits, alongside the existing learned relative-position bias.
The contact embeddings are frozen, detached and masked before the matrix
product; these are continuous dot products, not thresholded contacts or native
geometry. Two layers/eight heads add 16 trainable scalars, initialized to zero.
Zero initialization preserves identical initial control weights and outputs
under seed42; the new coefficients can then learn positive or negative weights.
The existing contact-dot scalar in the final pairwise head remains unchanged.

The control Transformer is still active and verifies source hashes, so this
variant is implemented in an isolated Python source snapshot at
`runs/se3_distance_contact_attention_20261023/code`, not by modifying its active
dependencies. The modified model/trainer and additional tests are preserved
there; integration into the main source tree is deferred until the control
finishes. Both implementation and training outputs persist independently.

The opt-in flag in the isolated trainer is `--contact-attention`, requiring
nonzero `--context-layers`. The default remains unchanged. Tests verify exact
zero-bias control parity, nonzero finite bias gradients, frozen input
gradients, dot-product influence, masking, target independence, symmetry,
sequence-control behavior and exact configuration/state reload. The CUDA
trainer smoke completed 32 updates with exact checkpoint reload and native-free
validation exports.

Full run: same 5,120/256 enriched cache, width256, two context layers/eight
heads, seed42, 512 pairs per chain, original optimizer/loss, 128 epochs or
12 elapsed training/evaluation hours. The final fixed-budget checkpoint is
used; compare equal updates with the control, not walltime alone. The
experiment starts fresh rather than fine-tuning control weights.
Output: `runs/se3_distance_contact_attention_20261023/training`.
Service: `foldtree2-distance-contact-attention-20261023.service`, GPU0,
13-hour service ceiling and 64-GiB host-memory ceiling. The control on GPU1
continues untouched. No production or coordinate-reconstruction changes.
