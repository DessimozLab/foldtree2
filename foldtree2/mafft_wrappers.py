#!/usr/bin/env python3
"""
Wrapper scripts for mafft_tools executables.
"""

import subprocess
import sys
import os
from pathlib import Path


def _get_executable_path(executable_name):
    """Get the path to the mafft_tools executable."""

    package_dir = Path(__file__).parent
    executable_path = package_dir / "mafft_tools" / executable_name
    
    if not executable_path.exists():
        print(f"Error: Executable {executable_name} not found at "
              f"{executable_path}")
        sys.exit(1)
    
    # Ensure the executable has proper permissions
    os.chmod(executable_path, 0o755)
    return executable_path


def _get_mad_executable_path(executable_name):
    """Get the path to the madroot executable."""
    package_dir = Path(__file__).parent
    executable_path = package_dir / "madroot" / executable_name
    
    if not executable_path.exists():
        print(f"Error: Executable {executable_name} not found at "
              f"{executable_path}")
        sys.exit(1)
    
    # Ensure the executable has proper permissions
    os.chmod(executable_path, 0o755)
    return executable_path


def _get_raxml_executable_path(executable_name):
    """Get the path to the raxml-ng executable."""
    package_dir = Path(__file__).parent
    executable_path = package_dir / "raxml-ng" / executable_name
    
    if not executable_path.exists():
        print(f"Error: Executable {executable_name} not found at "
              f"{executable_path}")
        sys.exit(1)
    
    # Ensure the executable has proper permissions
    os.chmod(executable_path, 0o755)
    return executable_path


def hex2maffttext_main():
    """Wrapper for hex2maffttext executable."""
    executable_path = _get_executable_path("hex2maffttext")
    
    try:
        result = subprocess.run([str(executable_path)] + sys.argv[1:],
                                check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"Error: Could not execute {executable_path}")
        sys.exit(1)


def maffttext2hex_main():
    """Wrapper for maffttext2hex executable."""
    executable_path = _get_executable_path("maffttext2hex")
    
    try:
        result = subprocess.run([str(executable_path)] + sys.argv[1:],
                                check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"Error: Could not execute {executable_path}")
        sys.exit(1)


def mad_main():
    """Wrapper for mad rooting executable."""
    executable_path = _get_mad_executable_path("mad")
    
    try:
        result = subprocess.run([str(executable_path)] + sys.argv[1:],
                                check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"Error: Could not execute {executable_path}")
        sys.exit(1)


def raxml_ng_main():
    """Wrapper for raxml-ng executable."""
    executable_path = _get_raxml_executable_path("raxml-ng")
    
    try:
        result = subprocess.run([str(executable_path)] + sys.argv[1:],
                                check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"Error: Could not execute {executable_path}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("This module provides wrapper functions for "
              "mafft_tools executables")
        sys.exit(1)
