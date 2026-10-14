# Clariden architecture comparison

The authorized comparison queues five four-epoch trials: control, continuous
pairs, learned seed, pairs plus seed, and pairs plus seed with two shared passes.
All start from the same original growth epoch-16 checkpoint and retain the same
original loss, 8,192/512 split, seed/order, LR, global batch, width/depth, precision
and optimizer reset. The separate completed loss trials do not change this
architecture comparison's objective or parent.

Two prerequisite jobs test **each** variant: one node / four GPUs for exact
save/resume versus uninterrupted training, then two nodes / eight GPUs for model,
optimizer and validation agreement at atol/rtol 1e-4. Both process two global
updates and validate all 512 structures. New zero-initialized pair/seed heads must
have finite nonzero updated weights. A failed gate stops downstream allocations.

| Stage | Nodes | GPUs | Maximum walltime | GPUh reservation |
|---|---:|---:|---:|---:|
| All five one-node recovery checks | 1 | 4 | 30 min | 2 |
| All five two-node agreement checks | 2 | 8 | 20 min | 2.667 |
| Each of five four-epoch trials | 2 | 8 | 45 min | 6 |
| Total | | | | **34.667** |

The normal partition, account `a0117`, established pinned extended container and
ARM training environment are reused. No automatic retries are enabled. The
previously faulty `nid007494` is excluded. Jobs run sequentially behind `afterok`
dependencies with invalid dependencies cancelled. Runtime check YAML files are
derived deterministically from canonical configs; only output and node count
change, and their hashes are included in submission receipts.

Before submission, the controller verifies the original parent SHA256
`93ff91d18ff9b4b0df5396889f9cd6c4fc0c91f65f88775caf71d2c90487bb04`,
cache/split manifest SHA256
`b2c635e8cf30af851a63eb8f0836ff090fb5707b51523a6b709cca8b707877f3`,
parent completion, fresh outputs, existing reservations and total rollout ceiling.
Every command receives `sbatch --test-only` before any stage is reserved/submitted.
The durable ledger prevents duplicate stages and retains reservations on failures.

The pre-existing ledger reserves 150 GPUh. This comparison brings it to
184.667 GPUh, below the original 256 GPUh authorization. Project credit uses the
user's earlier sufficient-credit confirmation; Slurm account association and
fair-share are not independently retrieved portal balances.

```bash
python3 -m scripts.submit_se3_architecture_trials --available-credit 34.66666666666667
python3 -m scripts.submit_se3_architecture_trials --available-credit 34.66666666666667 --submit
```

Run only from the pinned code archive on Clariden. Code, logs and generated check
configs live under `/capstor/store/cscs/swissai/a0117/foldtree2/se3/code/<revision>`.
Training outputs persist under `.../se3/runs/pairs_seed_v1_*`; gate reports and
submission receipts under `.../se3/artifacts/pairs_seed_v1_*`; budget reservations
under `.../se3/budget.jsonl`. The original frozen cache is reused without graph or
production checkpoint changes. The archive passes the 26-dependency release gate.

Local verification: 33 architecture/recycling/submission tests passed; prior full
architecture validation passed 105 tests with one GPU-only skip and one expected
historical depth-six failure. CPU production-cache gradients and exact starting
outputs are documented in [the architecture report](se3_pairs_seed_architecture.md).
Successful submission establishes a queue entry, not a passed CUDA gate or trained
quality improvement. No merge occurs automatically.

Cluster policy was rechecked against the live normal partition (12-hour maximum,
exclusive nodes) and [current Clariden documentation](https://docs.cscs.ch/clusters/clariden/).
GH200 cost uses four GPUh per node-hour under [MLP accounting](https://docs.cscs.ch/platforms/mlp/policies/).
Access and submission follow the local [Clariden skill](/home/dmoi/.codex/skills/clariden/SKILL.md).
