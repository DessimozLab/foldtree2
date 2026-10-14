# Training-data growth trial, 2026-10-14

The user requested more structures after the normalized medium model completed
the [16-epoch trial](se3_medium_fp64_trial_20261014.md) with stable finite training
and improving held-out geometry. This is an explicit training-data growth
experiment, independent of the still-unpassed matched model-capacity gate.
The historical `prepare_se3_expanded_cohort` workflow remains unchanged; its
new-validation and capacity-win protocol is a different experiment.

Use **8,192 training structures / the same 512 validation structures**, up to
256 residues per complete chain. Retain all existing 4,096 training IDs, add
4,096 fresh eligible IDs from the existing mk2 HDF5, and exclude every retained
holdout and the source cohort's earlier diagnostic IDs from additions. No graph
regeneration or production-model retraining. Preserve frozen extractor and
production checkpoint hashes, latent/codebook features, geometry contacts,
targets/masks, calibration and seed lift. Reuse existing cached entries byte for
byte and extract only the new entries. The source dataset has 20,090 structures;
eligibility is determined from residue count, confidence, finite coordinates,
defined frames and available angles, with skipped structures recorded.

Both workstation GPUs were occupied during preparation. Use CPU for frozen
extraction without interrupting either GPU workload; verify cache/uncached
parity and unchanged frozen weights/absence of gradients. CPU preparation is
not a local training run and spends no Clariden GPU allocation.

```bash
python -m scripts.prepare_se3_growth_cohort \
  --config configs/clariden_se3/prepare_growth_local.yaml
python -m scripts.export_se3_cache \
  --cohort runs/se3_growth_cache_20261014/cohort.json \
  --calibration runs/se3_loss_ablation_20261009/provenance.json \
  --output runs/se3_clariden/bundle_growth_8192
```

Training configuration: `configs/clariden_se3/growth_fp64_8192.yaml`. Keep the
previous width 64/depth 4/four-head model, input normalization, float64 gradient
accumulation/reduction, float32 forward/backward/AdamW, seed 42, global batch 16,
learning rate 5e-4, clipping at 1, losses, scheduler disabled and 16 epochs.
Start from the same fresh initialization rather than warm-starting the previous
trial, so training data is the intended change. This doubles the completed
optimizer updates to **8,192** and training visits to **131,072**; comparisons
must acknowledge both more unique structures and more optimization steps.

Use two GH200 nodes/eight GPUs on `normal`, at most two hours: **16 GPUh** ceiling.
The shared 256 GPUh ledger remains authoritative, moving from 68 to 84 GPUh
conservative reservations on submission. The user previously confirmed enough
credit; neither this ledger nor Slurm fair-share is a portal balance. The
previous measured 26m40s trial suggests approximately 50 minutes including
overheads; this is an estimate, not a guarantee. No auto-requeue or follow-on
expansion is enabled.

The bounded launcher checks earlier stability gates and completed learning,
retained split membership, preprocessing identity and entry hashes. After
staging a verified fresh revision directory and both durable and scratch cache:

```bash
python3 -m scripts.submit_se3_growth_trial --available-credit 256 --submit
```

Use the already verified pinned ARM image/venv and one torchrun launcher per
node/four workers each. Durable cache:
`/capstor/store/cscs/swissai/a0117/foldtree2/se3/cache/bundle_growth_8192`.
Scratch cache: `/iopsstor/scratch/cscs/dmoi/se3/bundle_growth_8192`.
Durable output: `/capstor/store/cscs/swissai/a0117/foldtree2/se3/runs/growth_fp64_8192`.
Save every 64 completed updates and every epoch, preserving optimizer, RNG,
stream and strict provenance. Select `best.ckpt` by complete-epoch validation
geometry and preserve every paired validation result.

Assess learning curves, finite model/optimizer states, complete sample counts,
paired FAPE, aligned CA RMSD, TM-score, lDDT, backbone geometry and angle errors
against both raw decoder and the previous medium trial on identical holdout
IDs. The repeatedly monitored holdout is not a new independent confirmation set;
production pretraining independence is not claimed. A material regression or
failure requires diagnosis rather than automatic resubmission or larger scale.
