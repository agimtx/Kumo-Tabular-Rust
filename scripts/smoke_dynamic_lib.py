#!/usr/bin/env python3
"""Load a packaged dynamic library and confirm its bundled ONNX Runtime works.

The check loads the packaged library, asks it to open a model that does not
exist, and expects the model-open error. Any failure to load the bundled ONNX
Runtime is surfaced by the library as a different error, so this distinguishes a
working bundle from one whose runtime cannot be loaded.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path


def find_packaged_library(package: Path) -> Path:
    lib_dir = package / "lib"
    candidates = [
        entry
        for entry in sorted(lib_dir.iterdir())
        if entry.is_file() and entry.name.startswith("libkumo-onnx-")
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one library in {lib_dir}, found {[str(path) for path in candidates]}"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()

    library = find_packaged_library(args.package)
    module = ctypes.CDLL(str(library))
    module.tabular_onnx_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    module.tabular_onnx_create.restype = ctypes.c_int32
    module.tabular_onnx_last_error.restype = ctypes.c_char_p

    handle = ctypes.c_void_p()
    status = module.tabular_onnx_create(b"missing-model.onnx", ctypes.byref(handle))
    error = (module.tabular_onnx_last_error() or b"").decode("utf-8", "replace")

    if status == 0:
        raise SystemExit(f"unexpected success loading a missing model with {library}")
    if handle.value:
        raise SystemExit("failed create must leave the handle NULL")
    if "does not exist" not in error:
        raise SystemExit(
            f"bundled ONNX Runtime did not load correctly; create failed with: {error}"
        )

    print(f"bundled ONNX Runtime loaded for {library.name}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())