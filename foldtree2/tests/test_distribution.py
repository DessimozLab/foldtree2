"""Archive and native-wrapper regressions without heavyweight model imports."""
from pathlib import Path
import sys
import os
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from check_distribution import RESOURCES, check_archive
from foldtree2 import mafft_wrappers


class DistributionTests(unittest.TestCase):
    def test_conda_smoke_solves_package_spec_instead_of_explicit_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "channel/linux-64/foldtree2-0.1.0-py310_0.conda"
            package.parent.mkdir(parents=True)
            package.touch()
            binary = root / "bin"
            binary.mkdir()
            log = root / "arguments"
            conda = binary / "conda"
            conda.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CALL_LOG"\nexit 17\n')
            conda.chmod(0o755)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/smoke_test_conda_cli.sh"),
                 "--package", str(package)],
                env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}",
                     "CALL_LOG": str(log)},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 17, result.stderr)
            arguments = log.read_text().splitlines()
            self.assertEqual(arguments[0], "create")
            self.assertIn("foldtree2=0.1.0=py310_0", arguments)
            self.assertIn(f"file://{root / 'channel'}", arguments)
            self.assertNotIn(str(package), arguments)

    def wheel(self, directory, extra=(), omit=()):
        path = Path(directory) / "foldtree2-0.1.0-py3-none-linux_x86_64.whl"
        with zipfile.ZipFile(path, "w") as archive:
            for name in (RESOURCES - set(omit)) | set(extra):
                archive.writestr(name, "fixture")
            archive.writestr(
                "foldtree2-0.1.0.dist-info/WHEEL",
                "Root-Is-Purelib: false\nTag: py3-none-linux_x86_64\n",
            )
        return path

    def test_valid_linux_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            check_archive(self.wheel(directory))

    def test_rejects_universal_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.wheel(directory)
            renamed = path.with_name("foldtree2-0.1.0-py3-none-any.whl")
            path.rename(renamed)
            with self.assertRaisesRegex(ValueError, "expected Linux"):
                check_archive(renamed)

    def test_missing_resource_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing"):
                check_archive(self.wheel(directory, omit={"foldtree2/madroot/mad"}))

    def test_release_rejects_unaudited_local_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "manylinux"):
                check_archive(self.wheel(directory), release=True)

    def test_release_accepts_manylinux_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "foldtree2-0.1.0-py3-none-manylinux_2_34_x86_64.whl"
            with zipfile.ZipFile(path, "w") as archive:
                for name in RESOURCES:
                    archive.writestr(name, "fixture")
                archive.writestr(
                    "foldtree2-0.1.0.dist-info/WHEEL",
                    "Root-Is-Purelib: false\nTag: py3-none-manylinux_2_34_x86_64\n",
                )
            check_archive(path, release=True)

    def test_research_data_and_scratch_modules_fail(self):
        for name in ("foldtree2/model.pt", "foldtree2/notebooks/demo.ipynb",
                     "foldtree2/tests/test_x.py", "foldtree2/scaling_experiment.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "unwanted"):
                    check_archive(self.wheel(directory, extra={name}))

    def test_sdist_resource_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "foldtree2-0.1.0.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                for name in RESOURCES:
                    member = tarfile.TarInfo(f"foldtree2-0.1.0/{name}")
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"x"))
            check_archive(path)

    def test_executable_wrappers_do_not_chmod_shared_installs(self):
        for resolver, tool in (
            (mafft_wrappers._get_executable_path, "hex2maffttext"),
            (mafft_wrappers._get_mad_executable_path, "mad"),
            (mafft_wrappers._get_raxml_executable_path, "raxml-ng"),
        ):
            with self.subTest(tool=tool), \
                    patch.object(mafft_wrappers.os, "access", return_value=True), \
                    patch.object(mafft_wrappers.os, "chmod") as chmod:
                self.assertTrue(resolver(tool).is_file())
                chmod.assert_not_called()


if __name__ == "__main__":
    unittest.main()
