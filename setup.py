"""Build hooks only; package metadata lives in pyproject.toml."""
import platform

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


class RuntimeBuild(build_py):
    """Exclude historical scratch/test modules from installed runtime code."""

    def find_package_modules(self, package, package_dir):
        return [
            item for item in super().find_package_modules(package, package_dir)
            if item[1] not in {"scaling_experiment", "test_chebconv_rework"}
        ]


class NativeDistribution(Distribution):
    """Install bundled native resources in the platform-specific library tree."""

    def has_ext_modules(self):
        return True


class LinuxWheel(bdist_wheel):
    """Bundled ELF executables make this a Linux wheel, not a universal wheel."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("FoldTree2 currently supports Linux x86_64 only")
        return "py3", "none", "linux_x86_64"


setup(distclass=NativeDistribution,
      cmdclass={"build_py": RuntimeBuild, "bdist_wheel": LinuxWheel})
