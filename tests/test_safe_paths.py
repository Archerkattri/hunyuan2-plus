"""Unit tests for safe_paths (stdlib only, no heavy dependencies)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safe_paths import (
    ALLOWED_MESH_TYPES,
    resolve_output_path,
    validated_mesh_type,
    validated_uid,
)


class ValidatedUidTests(unittest.TestCase):
    def test_canonical_uuid_roundtrip(self):
        uid = "12345678-1234-5678-1234-567812345678"
        self.assertEqual(validated_uid(uid), uid)

    def test_rejects_traversal_and_garbage(self):
        for bad in (
            "../secret",
            "..\\secret",
            "/etc/passwd",
            "abc/def.glb",
            "",
            "not-a-uuid",
            "1234",
            "12345678-1234-5678-1234-56781234567g",
        ):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    validated_uid(bad)


class ValidatedMeshTypeTests(unittest.TestCase):
    def test_allowlisted_types_pass_through(self):
        for ext in ALLOWED_MESH_TYPES:
            self.assertEqual(validated_mesh_type(ext), ext)

    def test_rejects_unknown_types(self):
        for bad in ("glb;rm", "../x", ".glb", "GLB", "", "exe", "glb\x00", "gltf2"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    validated_mesh_type(bad)


class ResolveOutputPathTests(unittest.TestCase):
    def test_stays_inside_save_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = resolve_output_path(tmp, "abc.glb")
            self.assertTrue(path.startswith(os.path.realpath(tmp) + os.sep))

    def test_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("../evil.glb", "../../etc/passwd", "/abs/path.glb", ""):
                with self.subTest(value=bad):
                    with self.assertRaises(ValueError):
                        resolve_output_path(tmp, bad)


if __name__ == "__main__":
    unittest.main()
