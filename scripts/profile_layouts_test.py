import json
from pathlib import Path
import tempfile
import unittest

from profile_layouts import MANIFEST, expand_arm, link_args


class LayoutTests(unittest.TestCase):
    def test_seed_zero_keeps_the_natural_layout(self):
        self.assertEqual(link_args(0), [])
        self.assertEqual(link_args(3), ["-Clink-arg=-Wl,--shuffle-sections=*=3"])

    def test_build_directory_expands_to_one_variant_per_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            layouts = [{"seed": seed, "binary": f"krasm-seed{seed}"} for seed in range(3)]
            (directory / MANIFEST).write_text(json.dumps({"kind": "layouts", "layouts": layouts}))
            variants, manifest = expand_arm("base", directory)
            self.assertEqual([label for label, _ in variants], ["base#0", "base#1", "base#2"])
            self.assertEqual(variants[2][1], directory / "krasm-seed2")
            self.assertEqual(manifest["kind"], "layouts")

    def test_lone_binary_is_a_group_of_one(self):
        with tempfile.NamedTemporaryFile() as binary:
            variants, manifest = expand_arm("one", binary.name)
            self.assertEqual(variants, [("one", Path(binary.name))])
            self.assertIsNone(manifest)

    def test_directory_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, MANIFEST):
                expand_arm("bad", temporary)


if __name__ == "__main__":
    unittest.main()
