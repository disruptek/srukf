#!/usr/bin/env python3
"""Setup script for the srukf Python package.

Handles building/bundling the C shared library during installation.
"""

import os
import platform
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py


def _lib_name():
    system = platform.system()
    if system == "Windows":
        return "srukf.dll"
    elif system == "Darwin":
        return "libsrukf.dylib"
    return "libsrukf.so"


def _repo_root():
    """Return the repository root (parent of python/)."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))


class BuildWithLibrary(build_py):
    """Custom build step that compiles libsrukf and bundles it."""

    def run(self):
        self._build_and_bundle()
        super().run()

    def _build_and_bundle(self):
        repo = _repo_root()
        lib = _lib_name()
        src = os.path.join(repo, lib)

        # Build if not already present
        if not os.path.isfile(src):
            print(f"Building {lib} via 'make lib'...")
            try:
                subprocess.check_call(["make", "lib"], cwd=repo)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(
                    f"Warning: could not build {lib}: {exc}\n"
                    f"The package will look for a system-installed library at runtime."
                )
                return

        # Copy into the package directory
        pkg_dir = os.path.join(self.build_lib, "srukf")
        os.makedirs(pkg_dir, exist_ok=True)
        dst = os.path.join(pkg_dir, lib)
        print(f"Bundling {src} -> {dst}")
        shutil.copy2(src, dst)


setup(
    cmdclass={"build_py": BuildWithLibrary},
)
