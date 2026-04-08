# Representation Conversion Guide (PDB ⇄ RT ⇄ Quaternion)

This guide documents the conversion flow used by
`foldtree2/scripts/test_representation_conversions.py`.

## Overview

The script follows this sequence:

1. **PDB backbone extraction**
   - Extract per-residue backbone atom coordinates: `N`, `CA`, `C`.
2. **Backbone coordinates → local frames**
   - Use `PDB2PyG.compute_local_frame(coords)` with `coords` of shape `(N, 3, 3)` in order `[N, CA, C]`.
   - Output:
     - `R`: rotation matrices, shape `(N, 3, 3)`
     - `t`: translation vectors, shape `(N, 3)`
3. **Rotation matrices → quaternions**
   - Use `rotation_matrix_to_quaternion(R)`.
   - Quaternion convention in this repo: `(w, x, y, z)` (scalar first).
4. **Quaternions → rotation matrices**
   - Use `quaternion_to_rotation_matrix(q)` for roundtrip reconstruction.
5. **RT → chain coordinates**
   - Use `reconstruct_positions(R, t)` to reconstruct coordinates from transforms.

## Noise Experiments

The script evaluates robustness by injecting noise in each representation:

- **Coordinate noise**: add Gaussian noise to `(N, CA, C)` coordinates, then recompute `R, t, q`.
- **RT noise**:
  - left-multiply random small rotations onto `R`
  - add Gaussian noise to `t`
- **Quaternion noise**:
  - add Gaussian noise to quaternion components
  - renormalize quaternions to unit norm
  - convert back to rotation matrices

## Losses

For each noisy variant, the script reports:

- **FAPE loss** via `fape_loss(true_R, true_t, pred_R, pred_t, batch)`
- **lDDT-style loss** via `compute_lddt_loss(true_positions, pred_positions)`
  where positions come from `reconstruct_positions`

Lower values indicate better consistency with the baseline representation.

## Run

From the repository root:

```bash
python -m foldtree2.scripts.test_representation_conversions \
  --pdb-path foldtree2/config/1eei.pdb \
  --coord-noise 0.25 \
  --rot-noise-rad 0.05 \
  --trans-noise 0.10 \
  --quat-noise 0.05 \
  --seed 0
```

## Key Notes

- Use backbone triplets `[N, CA, C]` to define local frames.
- CA-only coordinates are not sufficient for unique residue local orientation without extra assumptions.
- Keep quaternion convention consistent as `(w, x, y, z)` throughout conversions.
