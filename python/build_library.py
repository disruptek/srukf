#!/usr/bin/env python3
"""Build libsrukf and copy to the package directory.

Run this manually if you want to bundle the library without
going through ``pip install``:

    cd python
    python build_library.py
"""

import os
import platform
import shutil
import subprocess
import sys


def _lib_name():
    system = platform.system()
    if system == "Windows":
        return "srukf.dll"
    elif system == "Darwin":
        return "libsrukf.dylib"
    return "libsrukf.so"


def build():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.normpath(os.path.join(here, os.pardir))
    lib = _lib_name()

    print(f"Building {lib}...")
    subprocess.check_call(["make", "lib"], cwd=repo)

    src = os.path.join(repo, lib)
    dst = os.path.join(here, "srukf", lib)
    print(f"Copying {src} -> {dst}")
    shutil.copy2(src, dst)
    print("Done.")


if __name__ == "__main__":
    build()
