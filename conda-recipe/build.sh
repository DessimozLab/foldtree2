#!/bin/bash

# Install the package using pip (no dependencies - conda handles those)
$PYTHON -m pip install . --no-deps -vv

# Note: PyTorch Geometric should be installed separately by users
# via: pip install torch-geometric
# We don't install it here to reduce package size and avoid ZIP64 issues

# Copy external tools if needed
mkdir -p $PREFIX/bin
mkdir -p $PREFIX/share/foldtree2

# Copy bundled tools (raxml-ng, mad, mafft utilities)
if [ -d "raxml-ng" ]; then
    cp -r raxml-ng/* $PREFIX/share/foldtree2/
    if [ -f "$PREFIX/share/foldtree2/raxml-ng" ]; then
        chmod +x $PREFIX/share/foldtree2/raxml-ng
        ln -sf $PREFIX/share/foldtree2/raxml-ng $PREFIX/bin/raxml-ng
    fi
fi

if [ -d "madroot" ]; then
    cp -r madroot/* $PREFIX/share/foldtree2/
    if [ -f "$PREFIX/share/foldtree2/mad" ]; then
        chmod +x $PREFIX/share/foldtree2/mad
        ln -sf $PREFIX/share/foldtree2/mad $PREFIX/bin/mad
    fi
fi

if [ -d "mafft_tools" ]; then
    cp -r mafft_tools/* $PREFIX/share/foldtree2/
    if [ -f "$PREFIX/share/foldtree2/hex2maffttext" ]; then
        chmod +x $PREFIX/share/foldtree2/hex2maffttext
        ln -sf $PREFIX/share/foldtree2/hex2maffttext $PREFIX/bin/hex2maffttext
    fi
    if [ -f "$PREFIX/share/foldtree2/maffttext2hex" ]; then
        chmod +x $PREFIX/share/foldtree2/maffttext2hex
        ln -sf $PREFIX/share/foldtree2/maffttext2hex $PREFIX/bin/maffttext2hex
    fi
fi

# Copy configuration files
if [ -d "foldtree2/config" ]; then
    cp -r foldtree2/config $PREFIX/share/foldtree2/
fi

# Skip copying pretrained models to reduce package size
# Users should download models separately from GitHub releases
# echo "Note: Pretrained models not included in conda package to reduce size"
# echo "Download models from: https://github.com/DessimozLab/foldtree2/releases"
