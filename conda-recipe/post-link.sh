#!/bin/bash

cat << 'EOF'

╔═════════════════════════════════════════════════════════════╗
║              FoldTree2 Installation Complete!               ║
╚═════════════════════════════════════════════════════════════╝

⚠️  IMPORTANT: Additional setup required

1️⃣  Install PyTorch Geometric (required for graph neural networks):
   
   pip install torch-geometric

2️⃣  Download pretrained models (if needed):
   
   Visit: https://github.com/DessimozLab/foldtree2/releases
   
   Or train your own models using:
   python foldtree2/learn_lightning.py --config config.yaml

3️⃣  Test your installation:
   
   foldtree2 --about
   pdbs-to-graphs --help

📚 Documentation: https://github.com/DessimozLab/foldtree2
🐛 Issues: https://github.com/DessimozLab/foldtree2/issues

EOF
