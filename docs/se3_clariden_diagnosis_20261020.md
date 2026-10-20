# Clariden execution and quality diagnosis, 2026-10-20

The four-epoch width comparison completed successfully. Numerical execution is
reproducible; increasing width has not passed the quality screen. The later SSH
failure is a separate access problem, not evidence of a failed training job.

## Recorded results

These aggregates were read through authenticated SSH earlier on October 20.
They have been preserved from that successful query, not refreshed through the
currently failing connection. [Observed aggregates and provenance limitation](se3_clariden_evidence/scale_capacity_v3/completed_aggregates_20261020.json).

Both models trained from scratch on 8,192 structures for four epochs / 2,048
updates, using the same 512 development structures, loss, pair inputs and two
shared refinement passes. One-node exact-resume and two-node agreement gates
passed for both widths with maximum parameter/buffer difference zero. Both
training jobs exited successfully without recorded failure events.

| Width | Job | Walltime | Training geometry, first → fourth epoch | Best dev epoch | FAPE ↓ | Bond MAE Å ↓ | Bend MAE rad ↓ |
|---|---:|---|---|---:|---:|---:|---:|
| 64 | 3700114 | 24m46s | 4.3230 → 3.1054 | 3 | 0.926371 | 0.29524 | 0.40857 |
| 80 | 3700115 | 24m47s | 4.3373 → 3.1836 | 4 | 0.926102 | 0.36154 | 0.45975 |

At their FAPE-selected checkpoints, width 80 reduces mean geometry loss by only
0.315%, below the existing 1% capacity threshold. Its FAPE difference is
−0.000269, but bond error worsens 22.45% and bend error worsens 12.53%, both
outside the existing 5% regression bound. The aggregate thresholds therefore
already reject a capacity promotion. Paired confidence intervals and checkpoint
hashes remain pending retrieval; no statistical-significance claim is made.

The earlier bend/recycling winner (FAPE 0.920546) had a different, longer training
history. Comparing it directly with these fresh four-epoch models cannot isolate
width or a Clariden-versus-workstation effect. Keep that trained checkpoint as
the incumbent; do not replace it with the width-80 candidate.

## Loss behavior needs attention

Validation components from the same first/fourth epochs:

| Width | Bond loss | Distance loss | Smooth training FAPE | Total geometry |
|---|---|---|---|---|
| 64 | 0.7877 → 0.0615 | 1.2267 → 0.9118 | 1.8865 → 2.0109 | 4.0392 → 3.0348 |
| 80 | 0.8664 → 0.1000 | 1.2428 → 0.9830 | 1.8859 → 1.9532 | 4.1352 → 3.1006 |

The active geometry objective is distance + smooth training FAPE + bond +
0.5 × bend. For width 64, the bond-loss reduction alone explains about 72% of
the net validation-geometry decrease. Smooth training FAPE increases while the
clamped evaluation FAPE decreases slightly. This is an observed objective
tradeoff; aggregate values do not establish a defective FAPE implementation,
gradient conflict, or saturation fraction.

Before changing weights, measure the contributions on fixed real structures:
per-term coordinate/model gradient norms and cosine similarities, unclamped
frame-error distribution and fraction above the evaluation clamp, and metrics
by chain length/seed quality. Compare predicted coordinate geometry with target
frames; do not use angle-head accuracy alone as evidence of correct geometry.
Use the existing matched masks and fixed frozen production pair throughout.

The immediate training candidate remains the trained width-64 bend/recycling
winner. A controlled continuation or the already prepared old-versus-expanded
data comparison is more informative than another width increase. Use matched
updates, retain the development split, and reserve fresh confirmation for the
selected final checkpoint. Do not loosen numerical/provenance checks or launch
more training based solely on decreasing summed loss. The conservative budget
remains 237.666667 / 256 GPUh; no new allocation was submitted for this diagnosis.

## Corrected comparison tooling

`scripts/se3_capacity_gate.py` still required the superseded 16-epoch / 4,096
update schedule, preventing analysis of the actual four-epoch experiment. It
now derives completion from the checkpoint's declared epoch count, training
count and global batch. It checks all epoch sample counts, failure events and
artifacts, best/last provenance, paired IDs, raw baselines, residue counts and
finite metrics. It reports paired FAPE deltas and seeded structure-bootstrap
intervals while retaining the existing quality thresholds.

This changes analysis only; training source identities and old checkpoints are
untouched. Once access is restored, download the two completed run directories
and run the corrected analyzer locally:

```bash
python -m scripts.se3_capacity_gate /path/to/scale_capacity_v3r1_width64 \
  /path/to/scale_capacity_v3r1_width80 --output /path/to/capacity-comparison.json
```

## SSH diagnosis

The current certificate is usable until October 21, 20:50:31 UTC. A bounded
verbose connection confirms successful certificate authentication and command
execution on Ela. Forwarding to `clariden.alps.cscs.ch` reaches OpenSSH and the
server accepts the certificate offer, but the connection closes before client
confirmation of successful authentication or command execution. An IPv4 jump
route produces the same closure. DNS from Ela currently resolves the target to
`clariden-ln001a.cscs.ch` (172.28.30.144).

This narrows the failure to the Clariden login/authentication path. It does not
identify the server-side cause. Do not rotate working keys, switch MFA entries,
renew an already valid certificate repeatedly, change host-key trust, or claim
an announced outage. The configured Ela/ProxyJump route matches the
[official CSCS SSH instructions](https://docs.cscs.ch/access/ssh/).
Server-side authentication/session logs or CSCS support may be needed if the
closure persists. No support message has been sent.

Local verification: nine capacity-analysis tests pass, including declared
four-epoch completion, incomplete/failure rejection, baseline/provenance/ID
mismatches, invalid metric eligibility, nonfinite metrics, and rejection of a
small FAPE gain with bond regression. Six existing capacity tests also pass
(including finite two-pass learning for both widths). No training code changed.
