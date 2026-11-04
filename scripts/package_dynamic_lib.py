#!/usr/bin/env python3
"""Package a built cdylib with its public C header, ONNX Runtime, and notes."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

EXTENSIONS = {"linux": ".so", "windows": ".dll", "macos": ".dylib"}


def _is_onnxruntime_library(path: Path, platform: str) -> bool:
    name = path.name
    if platform == "windows":
        return name.startswith("onnxruntime") and name.endswith(".dll")
    prefix = "libonnxruntime"
    if platform == "macos":
        return name.startswith(prefix) and name.endswith(".dylib")
    return name.startswith(prefix) and (name.endswith(".so") or ".so." in name)


def _copy_onnxruntime(source: Path, destination: Path, platform: str) -> list[str]:
    """Copy ONNX Runtime shared libraries, preserving versioned symlinks."""
    if not source.is_dir():
        raise NotADirectoryError(f"ONNX Runtime library directory not found: {source}")

    copied: list[str] = []
    for entry in sorted(source.iterdir()):
        if not _is_onnxruntime_library(entry, platform):
            continue
        target = destination / entry.name
        if entry.is_symlink():
            link_target = os.readlink(entry)
            if target.is_symlink() or target.exists():
                target.unlink()
            os.symlink(link_target, target)
        else:
            shutil.copy2(entry, target)
        copied.append(entry.name)

    if not copied:
        raise FileNotFoundError(
            f"no ONNX Runtime shared library for {platform} found in {source}"
        )
    return copied


def _runtime_notes(platform: str, onnxruntime_version: str | None) -> str:
    version = onnxruntime_version or "unknown"
    if platform == "windows":
        library_name = "onnxruntime.dll"
    else:
        library_name = f"libonnxruntime{EXTENSIONS[platform]}"
    return (
        f"ONNX Runtime {version} is bundled in lib/{library_name} and is loaded\n"
        "dynamically by this library on first use (ort load-dynamic); it is not\n"
        "linked into the library.\n"
        "Keep the bundled lib/ layout intact so the runtime stays next to the\n"
        "library. Set ORT_DYLIB_PATH to override the bundled runtime with an\n"
        "external ONNX Runtime build.\n"
        "The ONNX Runtime license is included as ONNXRUNTIME_LICENSE.\n"
    )


def _external_runtime_notes() -> str:
    return (
        "ONNX Runtime is loaded dynamically by ort (load-dynamic); it is not linked "
        "into this library or copied into this package.\n"
        "The consuming application must provide a compatible ONNX Runtime shared "
        "library and make it discoverable by the platform loader (for example, "
        "PATH on Windows, LD_LIBRARY_PATH on Linux, or the normal dylib search "
        "paths on macOS) before calling the API.\n"
        "Do not link against a system ONNX Runtime library at build time.\n"
    )


def package_library(
    source: Path,
    header: Path,
    output_root: Path,
    platform: str,
    arch: str,
    onnxruntime_dir: Path | None = None,
    onnxruntime_version: str | None = None,
    onnxruntime_license: Path | None = None,
) -> Path:
    if platform not in EXTENSIONS:
        raise ValueError(f"unsupported platform: {platform}")
    if not source.is_file():
        raise FileNotFoundError(f"built dynamic library not found: {source}")
    if not header.is_file():
        raise FileNotFoundError(f"public header not found: {header}")
    if onnxruntime_dir is None and (onnxruntime_version or onnxruntime_license):
        raise ValueError("--onnxruntime-version/--onnxruntime-license require --onnxruntime-dir")
    if onnxruntime_license is not None and not onnxruntime_license.is_file():
        raise FileNotFoundError(f"ONNX Runtime license not found: {onnxruntime_license}")

    package_name = f"libonnx-{platform}-{arch}"
    package_dir = output_root / package_name
    include_dir = package_dir / "include"
    lib_dir = package_dir / "lib"
    include_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(header, include_dir / "onnx_ffi.h")
    shutil.copy2(source, lib_dir / f"{package_name}{EXTENSIONS[platform]}")

    runtime_line = "ONNX Runtime: external (provided by the host)\n"
    if onnxruntime_dir is not None:
        bundled = _copy_onnxruntime(onnxruntime_dir, lib_dir, platform)
        runtime_line = (
            f"ONNX Runtime: {onnxruntime_version or 'unknown'} (bundled in lib/: "
            f"{', '.join(bundled)})\n"
        )
        if onnxruntime_license is not None:
            shutil.copy2(onnxruntime_license, package_dir / "ONNXRUNTIME_LICENSE")

    (package_dir / "BUILD_INFO.txt").write_text(
        f"Package: {package_name}\n"
        f"Platform: {platform}\nArchitecture: {arch}\n"
        "Build: GitHub Actions release cdylib\n"
        f"{runtime_line}",
        encoding="utf-8",
    )
    notes = (
        _runtime_notes(platform, onnxruntime_version)
        if onnxruntime_dir is not None
        else _external_runtime_notes()
    )
    (package_dir / "RUNTIME_DEPS.txt").write_text(notes, encoding="utf-8")
    return package_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--header", type=Path, default=Path("include/onnx_ffi.h"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=sorted(EXTENSIONS), required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument(
        "--onnxruntime-dir",
        type=Path,
        help="directory containing the ONNX Runtime shared library to bundle",
    )
    parser.add_argument(
        "--onnxruntime-version",
        help="ONNX Runtime version recorded in BUILD_INFO.txt and RUNTIME_DEPS.txt",
    )
    parser.add_argument(
        "--onnxruntime-license",
        type=Path,
        help="ONNX Runtime LICENSE file copied into the package",
    )
    args = parser.parse_args()
    if args.onnxruntime_dir is None and (args.onnxruntime_version or args.onnxruntime_license):
        parser.error("--onnxruntime-version and --onnxruntime-license require --onnxruntime-dir")

    result = package_library(
        args.source,
        args.header,
        args.output,
        args.platform,
        args.arch,
        onnxruntime_dir=args.onnxruntime_dir,
        onnxruntime_version=args.onnxruntime_version,
        onnxruntime_license=args.onnxruntime_license,
    )
    print(result)


if __name__ == "__main__":
    main()