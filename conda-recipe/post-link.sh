#!/bin/bash

cat << 'EOF'

╔═════════════════════════════════════════════════════════════╗
║              FoldTree2 Installation Complete!               ║
╚═════════════════════════════════════════════════════════════╝

⚠️  IMPORTANT: Additional setup required

1️⃣  Obtain a trusted matching model bundle:

   Model files are not included. See the user guide for bundle requirements;
   do not assume every release provides model download assets.

   Or train your own models using:
   python -m foldtree2.learn_lightning --help

2️⃣  Test your installation:
   
   foldtree2 --about
   pdbs-to-graphs --help

📚 Documentation: https://github.com/DessimozLab/foldtree2
🐛 Issues: https://github.com/DessimozLab/foldtree2/issues

EOF
