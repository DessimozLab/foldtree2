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

## Preparation status

The complete 8,192/512 cohort has been selected and audited: all original
training IDs remain as the prefix, all 512 validation IDs remain identical,
all 8,704 IDs are unique, and additions exclude prior diagnostic IDs. Selection
recorded 4,970 skipped candidates. Eight targeted growth/exclusion tests passed;
all twelve training source files match the completed medium trial byte for byte.
Training settings differ only in training count and cache/output paths. The
growth release passed all 20 archive dependency checks and is staged under
revision `1d29587ba0a2a7b7a320e84461131476978210e9`, archive SHA-256
`62470025a44c346c9ff937bdc99119707e9e9a5d3d0af9106dfee1185b74c7cc`.

CPU preparation failed the existing cached-versus-uncached parity check on the
first retained structure: coordinate maximum difference 0.000775367, exceeding
`atol=rtol=1e-6`. No new frozen entries were published. The tolerance was not
relaxed and the old cache was not overwritten. Preserve the selected cohort and
failure log in `runs/se3_growth_cache_20261014` and
`runs/se3_clariden/prepare-growth-20261014.log`. An earlier selection-only CPU
attempt was interrupted to limit PyTorch threads to two; no cluster allocation
was involved.

Both workstation GPUs were occupied: GPU 0 by an ESM workload and GPU 1 by a
resident llama-server. No workload was stopped. GPU extraction is pending a
user-selected available device in the original local preparation attempt. **The growth training job has not been submitted,
and its 16 GPUh reservation has not been made.** The shared ledger remains at
68 GPUh. The frozen parity failure is a preparation issue, not a recurrence of
the refiner's earlier training instability.

## Clariden preparation replaces local extraction

The user clarified that the work should run on Clariden and requested
continuation. Frozen-output preparation is therefore moved to a bounded
Clariden GPU allocation too; no workstation GPU needs to be freed. The earlier
CPU failure remains preserved. A certificate-status check was briefly blocked
by the automatic approval service's usage limit; the later user-directed
continuation succeeded without changing credentials or bypassing review.

`scripts/export_se3_graph_inputs.py` reads the existing HDF5 and serializes the
unaltered graph tensors for the 4,096 additions and one retained parity probe.
It verifies the original dataset/checkpoint/extractor hashes and records each
graph file hash in a content-addressed export manifest. The full 39 GB HDF5
stays on the workstation; only the selected existing graphs and the frozen
11 MB encoder/decoder pair are transferred. This is a serialization of existing
graphs, not regeneration. The transfer archive and every graph/checkpoint are
hash-verified on Clariden.

`alps/se3/extract_growth.sbatch` allocates one GH200 node on `normal` for at most
30 minutes (**2 GPUh**). It runs the pinned image/verified venv with extraction-only
Gemmi 0.7.3 isolated in a separate dependency directory. It requires the retained
GPU cache parity check at the existing `atol=rtol=1e-6`, checks repeat extraction
of the first new structure, and verifies production parameters remain unchanged
with no gradients. It publishes a fresh durable 8,704-entry frozen bundle,
copies it to scratch and verifies both inventories/hashes. Retained entries
are copied byte for byte. New entries' ARM/PyTorch/CUDA producer details and
the source graph export identity are bound into the bundle manifest.

Only after all preparation checks pass, the host batch shell submits the exact
authorized `growth_fp64_8192` training trial once. A preparation failure stops
the chain; there is no silent parity relaxation or automatic retry. Both stages
use the shared budget ledger: 68 prior conservative GPUh + 2 preparation + 16
training = **86 GPUh**, within the user's 256 GPUh ceiling. This supersedes the
earlier workstation-device dependency. Training mathematics and the retained
validation protocol remain unchanged.
