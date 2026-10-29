# Revised SE(3) 512/128 pilot

This is the follow-up to the successful [bounded loss ablation](se3_loss_ablation_20261009.md).
The fixed lifted-seed/circular-angle model and soft-FAPE/local-bond objective
are reinitialized with seed 42 and trained on the original 512 pilot training
structures, with 128 disjoint validation structures, at most 384 residues.
Validation IDs were used in earlier diagnostics; they are development
validation, not an untouched test cohort or a production-pretraining holdout.

Configuration: [se3_revised_pilot.yaml](../configs/se3_revised_pilot.yaml).
Training is capped at 20 epochs, with validation clamped FAPE early stopping
(patience five, minimum improvement 0.0001) and best-checkpoint selection.
The existing 10.928414 calibration factor is retained from training-only
calibration; no validation targets contribute to seed construction or fitting.
Frozen epoch-40 caches are verified and reused without production forward
calls or graph regeneration. No production model enters the optimizer.

```bash
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
export PYTHONPATH=/tmp/foldtree2-se3-deps:$PWD
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m scripts.train_se3_revised_pilot --config configs/se3_revised_pilot.yaml
# After interruption, with matching configuration and implementation:
python -m scripts.train_se3_revised_pilot --config configs/se3_revised_pilot.yaml --resume-from runs/se3_revised_pilot_20261009/last.pt
```

Run GPU commands outside a sandbox that hides CUDA device nodes, and only when
GPU 0 is available. GPU 1's unrelated workload is excluded from this process.
Logs and source snapshots are in `runs/se3_revised_pilot_20261009`.
`events.jsonl` records progress every 64 structures, epoch losses, gradient
norms, effective counts, throughput timing, and peak host/GPU memory.
`last.pt` and `best.pt` are written atomically. Resume restores optimizer,
scheduler, epoch/global step and CPU/CUDA RNG; changed dataset, splits,
calibration, configuration or source hashes are rejected.

On completion, `heldout_pairs.json` contains identical-mask raw-seed,
lifted-input and refined metrics, including clamped FAPE, aligned CA RMSD,
TM-align TM-score, CA lDDT, bond geometry and angle errors. Compare against both
raw and lifted inputs, and apply the existing material-regression thresholds
before considering larger training.

## Completed results

The pilot completed on 2026-10-12, stopping after 14 epochs when validation
patience reached five. The selected checkpoint is epoch nine. All 128 validation
structures were evaluated; FAPE improved on 118 of them. There were no recorded
training failure events or production model forward calls. The earlier process
interruption and recovery are described below.

| Metric | Lifted input | Refined |
|---|---:|---:|
| Clamped CA-frame FAPE | 0.947859 | 0.941036 |
| Aligned CA RMSD, Å | 18.403283 | 17.914139 |
| TM-score | 0.192221 | 0.206922 |
| CA lDDT | 0.195812 | 0.247108 |
| CA bond MAE, Å | 1.762846 | 1.291629 |
| CA bend MAE, radians | 0.843561 | 0.736896 |
| CA torsion MAE, radians | 1.566216 | 1.547142 |

All reported structural means improved over identical lifted inputs and masks.
The pilot passes the existing FAPE-improvement and no-material-regression gates
(>5% error increase or >0.01 TM-score/lDDT decrease). Together with the prior
21.36% bounded own-objective learning reduction, this supports preparing the
Clariden migration and a controlled larger experiment. Absolute reconstruction
quality remains limited; this is development validation reused in diagnostics,
not an independent generalization or production-readiness claim.

Results, curves, paired raw/input/refined measurements and executed source
snapshots are retained in `docs/se3_revised_pilot_evidence`; checkpoint hashes
identify local weights without replacing production checkpoints.

## Status on 2026-10-12

The initial process stopped after eight completed epochs (512 optimizer
updates), without a completion or error event. It had not reached the
20-epoch cap or early stopping; patience used was two. Training geometry
fell from 4.987154 after epoch one to 4.322217 after epoch eight (13.33%).
Best validation clamped FAPE was 0.941676 at epoch six, versus 0.944932 after
epoch one. These are partial results, not a passed structural gate.

On 2026-10-12, source hashes were verified and `last.pt` resumed on idle GPU 0
at epoch nine / global step 512, restoring optimizer, scheduler and RNG state.
The run is now supervised by the persistent user service
`foldtree2-se3-revised-pilot-20261012.service` rather than a tool execution
session. The existing epoch cap, early stopping and splits were retained.

```bash
systemctl --user status foldtree2-se3-revised-pilot-20261012.service
tail -n 10 runs/se3_revised_pilot_20261009/console.log
```

The launcher is `runs/se3_revised_pilot_20261009/resume.sh`. An initial
service launch failed because conda activation is incompatible with shell
`nounset`; the launcher was corrected and verified active. This launch failure
occurred before training and did not alter the checkpoints. At the time of
resume, final paired metrics were pending; completed results are reported above.
