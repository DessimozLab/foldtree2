# Contact dot-product experiments

The matched large-data screen uses 9,988 cached AFDB-derived training chains, 256 development-validation chains, four epochs, and seed 42. Both experimental heads use a 64-dimensional sphere, a learned global scale/bias, and contact-loss weight 0.3. The final-epoch checkpoints are compared with the earlier baseline and shared-sphere runs on exactly the same split.

- **Pair bias:** `scale * cosine(u_i, u_j) + global_bias + pair_mlp(i, j)`; the symmetric pair MLP is initialized to zero and uses the same frozen input features as the distance head.
- **Cross dot:** `scale * (dot(q_i, k_j) + dot(q_j, k_i)) / 2 + global_bias`, with normalized query/key projections and a small deterministic initial perturbation. The first cross-dot implementation initialized both projections identically, so they remained tied; those earlier results are marked invalid in MLflow.

The [four-model comparison](../runs/contact_dot_large_comparison_20261027/report.md) contains pooled and per-chain AP, threshold metrics, distance error, and learning curves. The [operating-point report](../runs/contact_dot_large_calibration_20261027/report.md) splits validation chains into 128 calibration and 128 held-out chains. Thresholds are chosen for 20% recall on calibration chains and measured on held-out chains. Both subsets come from a development holdout reused in earlier experiments; independence from production pretraining is unverified.

## Use the pair-bias checkpoint

The trained checkpoint is `runs/contact_objective_pair_bias_large_20261026/sphere64_w0p3/last.pt`. It is also attached to the [pair-bias MLflow run](http://172.26.141.223:5000/#/experiments/5/runs/f972454e68e24466b069594080b7f738). `entry` below is one frozen-feature record with `features`, `contact_z`, `aa_probs`, and a boolean `mask`; it does not require native target coordinates. All tensors must be on the same device.

```python
import json
import torch
from pathlib import Path
from foldtree2.src.distance.contact_dot_experiments import PairAdjustedDotHead
from foldtree2.src.distance.distance_restraints import eligible_pairs

checkpoint = torch.load(
    'runs/contact_objective_pair_bias_large_20261026/sphere64_w0p3/last.pt',
    map_location='cpu', weights_only=False,
)
model = PairAdjustedDotHead(**checkpoint['config'])
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Supply an entry produced by the frozen-feature pipeline.
pairs = eligible_pairs(entry['mask'], min_separation=12)
with torch.no_grad():
    distance_distribution = model(entry, pairs).softmax(-1)
    contact_scores = model.contact_probability(distance_distribution)

calibration = json.loads(Path('runs/contact_dot_large_calibration_20261027/calibration.json').read_text())
case = next(row for row in calibration['results'] if row['case'] == 'sphere64_w0p3_pair_bias')
threshold = case['calibration']['threshold']
predicted_contact_pairs = pairs[contact_scores >= threshold]
```

The empirically selected threshold is approximately **0.4534**. On the separate 128-chain held-out portion of the development split, it yielded 8.3% precision and 21.3% recall. Selecting pairs using this threshold leaves the model's distance-bin distribution unchanged. The scores are affected by balanced contact supervision and should not be assumed to be calibrated probabilities.

An earlier [experimental shifted shared-sphere checkpoint](../runs/contact_bias_calibration_20261026/sphere64_experimental_operating_point.pt) makes its selected cutoff correspond to 0.5 by changing the global logit bias. This also changes its complete distance distribution. Applying a threshold to the original scores is preferable when the original distance probabilities are needed.

## Reproduce the assessment

```bash
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -q foldtree2/tests/test_contact_dot_experiments.py foldtree2/tests/test_contact_residue_bias.py foldtree2/tests/test_contact_objectives.py
```

The scorer implementations are in `foldtree2/src/distance/contact_dot_experiments.py` and `foldtree2/src/distance/contact_cross_dot_perturbed.py`. Each experiment directory contains its source snapshot, provenance, training history, final validation metrics, and checkpoint. The local MLflow geometry watcher records training progress; comparison plots, calibration reports, and new checkpoints are attached to each corresponding run.
