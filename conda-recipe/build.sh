#!/bin/bash
set -euo pipefail

# Pip installs the same entry points and package resources as the PyPI wheel.
"$PYTHON" -m pip install . --no-deps --no-build-isolation -vv
