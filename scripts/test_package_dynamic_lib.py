import tempfile
import unittest
from pathlib import Path

from package_dynamic_lib import package_library


class PackageDynamicLibTests(unittest.TestCase):
    def test_creates_expected_layout_and_runtime_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "built-library"
            header = root / "onnx_ffi.h"
            source.write_bytes(b"library")
            header.write_text("/* public api */\n", encoding="utf-8")

            package = package_library(source, header, root / "out", "linux", "x86_64")

            self.assertEqual(package.name, "libonnx-linux-x86_64")
            self.assertEqual((package / "include/onnx_ffi.h").read_text(), "/* public api */\n")
            self.assertEqual((package / "lib/libonnx-linux-x86_64.so").read_bytes(), b"library")
            self.assertTrue((package / "BUILD_INFO.txt").is_file())
            deps = (package / "RUNTIME_DEPS.txt").read_text()
            self.assertIn("loaded dynamically", deps)
            self.assertIn("not linked", deps)

    def test_missing_header_fails_instead_of_packaging_incomplete_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "built-library"
            source.touch()
            with self.assertRaises(FileNotFoundError):
                package_library(source, root / "missing.h", root / "out", "windows", "x86_64")


if __name__ == "__main__":
    unittest.main()
