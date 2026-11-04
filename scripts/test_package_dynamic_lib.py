import tempfile
import unittest
from pathlib import Path

from package_dynamic_lib import package_library


class PackageDynamicLibTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.source = self.root / "built-library"
        self.header = self.root / "onnx_ffi.h"
        self.source.write_bytes(b"library")
        self.header.write_text("/* public api */\n", encoding="utf-8")

    def test_creates_expected_layout_and_runtime_notes(self):
        package = package_library(self.source, self.header, self.root / "out", "linux", "x86_64")

        self.assertEqual(package.name, "libonnx-linux-x86_64")
        self.assertEqual((package / "include/onnx_ffi.h").read_text(), "/* public api */\n")
        self.assertEqual((package / "lib/libaqonnx-x86_64.so").read_bytes(), b"library")
        self.assertTrue((package / "BUILD_INFO.txt").is_file())
        self.assertIn("Library: lib/libaqonnx-x86_64.so", (package / "BUILD_INFO.txt").read_text())
        deps = (package / "RUNTIME_DEPS.txt").read_text()
        self.assertIn("loaded dynamically", deps)
        self.assertIn("not linked", deps)
        self.assertIn("external", (package / "BUILD_INFO.txt").read_text())

    def test_names_library_after_arch_with_platform_extension(self):
        cases = (
            ("linux", "x86_64", "libaqonnx-x86_64.so"),
            ("macos", "aarch64", "libaqonnx-aarch64.dylib"),
            ("windows", "x86_64", "libaqonnx-x86_64.dll"),
        )
        for platform, arch, library_name in cases:
            with self.subTest(platform=platform):
                package = package_library(
                    self.source, self.header, self.root / "out", platform, arch
                )
                self.assertEqual(
                    [entry.name for entry in (package / "lib").iterdir() if entry.is_file()],
                    [library_name],
                )

    def test_missing_header_fails_instead_of_packaging_incomplete_output(self):
        with self.assertRaises(FileNotFoundError):
            package_library(self.source, self.root / "missing.h", self.root / "out", "windows", "x86_64")

    def test_bundles_single_onnxruntime_under_canonical_name(self):
        runtime_dir = self.root / "runtime-lib"
        runtime_dir.mkdir()
        (runtime_dir / "libonnxruntime.so.1.23.0").write_bytes(b"onnxruntime")
        (runtime_dir / "libonnxruntime.so").symlink_to("libonnxruntime.so.1.23.0")
        (runtime_dir / "unrelated.txt").write_text("ignore me", encoding="utf-8")
        license_file = self.root / "LICENSE"
        license_file.write_text("MIT", encoding="utf-8")

        package = package_library(
            self.source,
            self.header,
            self.root / "out",
            "linux",
            "x86_64",
            onnxruntime_dir=runtime_dir,
            onnxruntime_version="1.23.0",
            onnxruntime_license=license_file,
        )

        runtime = package / "lib/libonnxruntime.so"
        self.assertEqual(runtime.read_bytes(), b"onnxruntime")
        self.assertFalse(runtime.is_symlink())
        self.assertFalse((package / "lib/libonnxruntime.so.1.23.0").exists())
        self.assertFalse((package / "lib/unrelated.txt").exists())
        self.assertEqual(
            sorted(
                entry.name
                for entry in (package / "lib").iterdir()
                if "onnxruntime" in entry.name
            ),
            ["libonnxruntime.so"],
        )
        self.assertTrue((package / "ONNXRUNTIME_LICENSE").is_file())
        deps = (package / "RUNTIME_DEPS.txt").read_text()
        self.assertIn("ONNX Runtime 1.23.0 is bundled in lib/libonnxruntime.so", deps)
        self.assertIn("ORT_DYLIB_PATH", deps)
        info = (package / "BUILD_INFO.txt").read_text()
        self.assertIn("ONNX Runtime: 1.23.0", info)
        self.assertIn("libonnxruntime.so", info)

    def test_renames_versioned_runtime_when_only_that_file_is_present(self):
        runtime_dir = self.root / "runtime-versioned-only"
        runtime_dir.mkdir()
        (runtime_dir / "libonnxruntime.1.23.0.dylib").write_bytes(b"onnxruntime")

        package = package_library(
            self.source,
            self.header,
            self.root / "out",
            "macos",
            "aarch64",
            onnxruntime_dir=runtime_dir,
            onnxruntime_version="1.23.0",
        )

        self.assertEqual((package / "lib/libonnxruntime.dylib").read_bytes(), b"onnxruntime")
        self.assertFalse((package / "lib/libonnxruntime.1.23.0.dylib").exists())

    def test_bundled_layout_matches_requested_platform_name(self):
        runtime_dir = self.root / "runtime-win"
        runtime_dir.mkdir()
        (runtime_dir / "onnxruntime.dll").write_bytes(b"onnxruntime")

        package = package_library(
            self.source,
            self.header,
            self.root / "out",
            "windows",
            "x86_64",
            onnxruntime_dir=runtime_dir,
            onnxruntime_version="1.23.0",
        )

        self.assertEqual((package / "lib/onnxruntime.dll").read_bytes(), b"onnxruntime")
        self.assertIn("onnxruntime.dll", (package / "RUNTIME_DEPS.txt").read_text())

    def test_rejects_empty_onnxruntime_directory(self):
        runtime_dir = self.root / "empty"
        runtime_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            package_library(
                self.source,
                self.header,
                self.root / "out",
                "linux",
                "x86_64",
                onnxruntime_dir=runtime_dir,
            )

    def test_version_without_runtime_directory_is_rejected(self):
        with self.assertRaises(ValueError):
            package_library(
                self.source,
                self.header,
                self.root / "out",
                "linux",
                "x86_64",
                onnxruntime_version="1.23.0",
            )


if __name__ == "__main__":
    unittest.main()