# Project Agent Instructions

## CUDA execution

The `foldtree2` conda environment is CUDA-capable, but Codex's restricted command sandbox may hide the NVIDIA device nodes. Before concluding that CUDA is unavailable, run GPU training and CUDA checks outside the restricted sandbox. Use:

```bash
source /home/dmoi/miniforge3/etc/profile.d/conda.sh
conda activate foldtree2
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

For multi-GPU Lightning runs, use the CUDA accelerator with the requested device count and DDP strategy, for example `--accelerator cuda --devices 2 --strategy ddp_find_unused_parameters_true`.
