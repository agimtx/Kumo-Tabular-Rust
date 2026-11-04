#!/usr/bin/env python3
"""Package a built cdylib with its public C header and runtime notes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EXTENSIONS = {"linux": ".so", "windows": ".dll", "macos": ".dylib"}


def package_library(
    source: Path, header: Path, output_root: Path, platform: str, arch: str
) -> Path:
    if platform not in EXTENSIONS:
        raise ValueError(f"unsupported platform: {platform}")
    if not source.is_file():
        raise FileNotFoundError(f"built dynamic library not found: {source}")
    if not header.is_file():
        raise FileNotFoundError(f"public header not found: {header}")

    package_name = f"libonnx-{platform}-{arch}"
    package_dir = output_root / package_name
    include_dir = package_dir / "include"
    lib_dir = package_dir / "lib"
    include_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(header, include_dir / "onnx_ffi.h")
    shutil.copy2(source, lib_dir / f"{package_name}{EXTENSIONS[platform]}")
    (package_dir / "BUILD_INFO.txt").write_text(
        f"Package: {package_name}\n"
        f"Platform: {platform}\nArchitecture: {arch}\n"
        "Build: GitHub Actions release cdylib\n",
        encoding="utf-8",
    )
    (package_dir / "RUNTIME_DEPS.txt").write_text(
        "ONNX Runtime is loaded dynamically by ort (load-dynamic); it is not linked "
        "into this library or copied into this package.\n"
        "The consuming application must provide a compatible ONNX Runtime shared "
        "library and make it discoverable by the platform loader (for example, "
        "PATH on Windows, LD_LIBRARY_PATH on Linux, or the normal dylib search "
        "paths on macOS) before calling the API.\n"
        "Do not link against a system ONNX Runtime library at build time.\n",
        encoding="utf-8",
    )
    return package_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--header", type=Path, default=Path("include/onnx_ffi.h"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=sorted(EXTENSIONS), required=True)
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()
    result = package_library(args.source, args.header, args.output, args.platform, args.arch)
    print(result)


if __name__ == "__main__":
    main()
