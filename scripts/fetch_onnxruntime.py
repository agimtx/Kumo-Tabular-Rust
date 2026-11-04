#!/usr/bin/env python3
"""Download and extract a pinned ONNX Runtime release for a target platform."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# ONNX Runtime release assets used by the dynamic-library workflow.
ASSETS = {
    ("linux", "x86_64"): ("onnxruntime-linux-x64", "tgz"),
    ("windows", "x86_64"): ("onnxruntime-win-x64", "zip"),
    ("macos", "aarch64"): ("onnxruntime-osx-arm64", "tgz"),
}

RELEASE_URL = (
    "https://github.com/microsoft/onnxruntime/releases/download/"
    "v{version}/{archive}"
)


def fetch_onnxruntime(version: str, platform: str, arch: str, output: Path) -> Path:
    key = (platform, arch)
    if key not in ASSETS:
        raise ValueError(f"no ONNX Runtime asset for platform={platform} arch={arch}")
    base_name, extension = ASSETS[key]
    root_name = f"{base_name}-{version}"
    archive = output / f"{root_name}.{extension}"
    output.mkdir(parents=True, exist_ok=True)

    url = RELEASE_URL.format(version=version, archive=archive.name)
    with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    if extension == "zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(output)
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            try:
                bundle.extractall(output, filter="data")
            except TypeError:  # Python < 3.11.4 has no extraction filter
                bundle.extractall(output)

    extracted = output / root_name
    if not extracted.is_dir():
        raise FileNotFoundError(f"ONNX Runtime archive did not contain {root_name}")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="ONNX Runtime version, e.g. 1.23.0")
    parser.add_argument("--platform", required=True, choices=["linux", "windows", "macos"])
    parser.add_argument("--arch", required=True, help="x86_64 or aarch64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = fetch_onnxruntime(args.version, args.platform, args.arch, args.output)
    print(root)


if __name__ == "__main__":
    main()