#!/bin/bash
# Post-install script to install packages only available via pip

echo "Installing PyTorch Geometric and pydssp via pip..."
$PYTHON -m pip install --no-deps torch-geometric pydssp transformers
echo "Post-install complete!"
