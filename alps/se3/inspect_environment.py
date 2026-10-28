"""Run once inside an allocated container, never on a login node."""
import importlib.metadata as metadata
import json
import platform
import torch
from foldtree2.train_se3_lightning import build_model
names = ['pytorch-lightning', 'gotennet-pytorch', 'e3nn', 'torch-geometric', 'einops', 'numpy', 'tmtools']
report = {'architecture': platform.machine(), 'python': platform.python_version(),
          'torch': torch.__version__, 'cuda': torch.version.cuda,
          'packages': {n: metadata.version(n) for n in names},
          'gpu_count': torch.cuda.device_count()}
print(json.dumps(report, indent=2), flush=True)
expected = {'pytorch-lightning':'2.6.0', 'gotennet-pytorch':'0.3.1', 'e3nn':'0.5.9',
            'torch-geometric':'2.6.1', 'einops':'0.8.1', 'numpy':'1.26.4'}
if any(report['packages'][k] != v for k,v in expected.items()):
    raise RuntimeError('Installed packages differ from validated pins')
if platform.machine() != 'aarch64' or not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise RuntimeError('Expected ARM GH200 allocation with four CUDA devices')
for width, depth, heads in [(32, 2, 2), (64, 4, 4)]:
    model = build_model(16, {'hidden': width, 'depth': depth, 'heads': heads}).cuda()
    print(json.dumps({'width': width, 'parameters': sum(p.numel() for p in model.parameters())}))
