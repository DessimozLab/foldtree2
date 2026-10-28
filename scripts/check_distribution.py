#!/usr/bin/env python3
"""Reject missing resources, universal wheels, or research data in pip archives."""
import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

RESOURCES = {
    "foldtree2/__init__.py",
    "foldtree2/src/encoder.py",
    "foldtree2/src/pdbgraphmk2.py",
    "foldtree2/config/aaindex1.csv",
    "foldtree2/config/3diphy_submats/Q.3Di.AF",
    "foldtree2/raxml-ng/raxml-ng",
    "foldtree2/raxml-ng/LICENSE.txt",
    "foldtree2/madroot/mad",
    "foldtree2/mafft_tools/hex2maffttext",
    "foldtree2/mafft_tools/maffttext2hex",
}
FORBIDDEN_SUFFIXES = {".h5", ".hdf5", ".ipynb", ".pt", ".pth", ".pkl", ".pyc"}
FORBIDDEN_DIRS = {"notebooks", "tests", "models", "runs", "families", "__pycache__"}


def check_archive(path, release=False):
    """Inspect without extracting; require the same resources in sdist and wheel."""
    path = Path(path)
    if path.stat().st_size > 50 * 1024**2:
        raise ValueError(f"{path}: exceeds 50 MiB")
    if path.suffix == ".whl":
        tags = ["manylinux_2_34_x86_64"] if release else [
            "linux_x86_64", "manylinux_2_34_x86_64",
        ]
        tag = next((tag for tag in tags if path.name.endswith(f"-py3-none-{tag}.whl")), None)
        if tag is None:
            raise ValueError(f"{path}: expected Linux x86_64 wheel with platform in {tags}")
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            wheels = [n for n in names if n.endswith(".dist-info/WHEEL")]
            if len(wheels) != 1:
                raise ValueError(f"{path}: expected one WHEEL metadata file")
            wheel = archive.read(wheels[0]).decode()
            if "Root-Is-Purelib: false" not in wheel or f"Tag: py3-none-{tag}" not in wheel:
                raise ValueError(f"{path}: incorrect native platform metadata")
    else:
        with tarfile.open(path) as archive:
            names = {n.name.split("/", 1)[1] for n in archive.getmembers()
                     if n.isfile() and "/" in n.name}
    missing = RESOURCES - names
    unwanted = sorted(n for n in names if (
        PurePosixPath(n).suffix in FORBIDDEN_SUFFIXES
        or FORBIDDEN_DIRS.intersection(PurePosixPath(n).parts)
        or n.endswith(("/scaling_experiment.py", "/test_chebconv_rework.py"))
    ))
    if missing or unwanted:
        raise ValueError(f"{path}: missing={sorted(missing)}, unwanted={unwanted}")
    print(f"{path.name}: resource, platform and data-exclusion checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true",
                        help="Require the audited manylinux release tag, not a local Linux wheel")
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.archives:
        check_archive(path, release=args.release)


if __name__ == "__main__":
    main()
