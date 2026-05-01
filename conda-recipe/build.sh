#!/bin/bash
set -euo pipefail

# Install the package using pip (no dependencies - conda handles those)
$PYTHON -m pip install . --no-deps -vv

# Note: PyTorch Geometric should be installed separately by users
# via: pip install torch-geometric
# We don't install it here to reduce package size and avoid ZIP64 issues

# Copy external tools and runtime resources.
mkdir -p $PREFIX/bin
mkdir -p $PREFIX/share/foldtree2

# Copy bundled tools from package source tree.
if [ -d "foldtree2/raxml-ng" ]; then
    cp -r foldtree2/raxml-ng "$PREFIX/share/foldtree2/"
    if [ -f "$PREFIX/share/foldtree2/raxml-ng/raxml-ng" ]; then
        chmod +x "$PREFIX/share/foldtree2/raxml-ng/raxml-ng"
        ln -sf "$PREFIX/share/foldtree2/raxml-ng/raxml-ng" "$PREFIX/bin/raxml-ng"
    fi
fi

if [ -d "foldtree2/madroot" ]; then
    cp -r foldtree2/madroot "$PREFIX/share/foldtree2/"
    if [ -f "$PREFIX/share/foldtree2/madroot/mad" ]; then
        chmod +x "$PREFIX/share/foldtree2/madroot/mad"
        ln -sf "$PREFIX/share/foldtree2/madroot/mad" "$PREFIX/bin/mad"
    fi
fi

if [ -d "foldtree2/mafft_tools" ]; then
    cp -r foldtree2/mafft_tools "$PREFIX/share/foldtree2/"
    if [ -f "$PREFIX/share/foldtree2/mafft_tools/hex2maffttext" ]; then
        chmod +x "$PREFIX/share/foldtree2/mafft_tools/hex2maffttext"
        ln -sf "$PREFIX/share/foldtree2/mafft_tools/hex2maffttext" "$PREFIX/bin/hex2maffttext"
    fi
    if [ -f "$PREFIX/share/foldtree2/mafft_tools/maffttext2hex" ]; then
        chmod +x "$PREFIX/share/foldtree2/mafft_tools/maffttext2hex"
        ln -sf "$PREFIX/share/foldtree2/mafft_tools/maffttext2hex" "$PREFIX/bin/maffttext2hex"
    fi
fi

# Copy configuration files
if [ -d "foldtree2/config" ]; then
    cp -r foldtree2/config $PREFIX/share/foldtree2/
fi

# Always create production models directory; copy files when present.
mkdir -p "$PREFIX/share/foldtree2/models/production"
if [ -d "models/production" ]; then
    cp -r models/production/. "$PREFIX/share/foldtree2/models/production/"
fi
