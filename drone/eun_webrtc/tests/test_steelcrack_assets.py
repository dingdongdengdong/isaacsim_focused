from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_steelcrack_assets import prepare_assets  # noqa: E402


class SteelCrackAssetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        for directory in (root / "Train/images", root / "Train/masks"):
            directory.mkdir(parents=True)
        for index in range(4):
            source_id = f"{index:06d}"
            Image.new("RGB", (512, 512), (80 + index, 90, 100)).save(
                root / "Train/images" / f"{source_id}.png"
            )
            mask = Image.new("L", (512, 512), 0)
            for x in range(100, 102 + index):
                mask.putpixel((x, 200), 255)
            mask.save(root / "Train/masks" / f"{source_id}.png")

    def test_prepares_rgba_from_train_only_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "runtime"
            self._fixture(source)
            result = prepare_assets(
                source,
                output,
                source_id="000000",
                selection_count=4,
                dilation_pixels=2,
                feather_radius=1.0,
            )

            provenance = result["provenance"]
            self.assertEqual(provenance["dataset"], "SteelCrack")
            self.assertEqual(provenance["source_split"], "Train")
            self.assertEqual(provenance["source_id"], "000000")
            self.assertEqual(len(provenance["texture_sha256"]), 64)
            texture = Image.open(provenance["texture"])
            self.assertEqual(texture.mode, "RGBA")
            self.assertEqual(texture.size, (512, 512))
            alpha = texture.getchannel("A")
            self.assertGreater(sum(alpha.histogram()[1:]), provenance["crack_pixels"])
            self.assertTrue(any(alpha.histogram()[value] for value in range(1, 255)))

            selection = json.loads((output / "selected_sources.json").read_text())
            self.assertEqual(selection["selection_count"], 4)
            self.assertTrue(all("/Train/" in item["image_path"] for item in selection["sources"]))

    def test_rejects_non_binary_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            self._fixture(source)
            mask_path = source / "Train/masks/000000.png"
            mask = Image.open(mask_path).convert("L")
            mask.putpixel((0, 0), 127)
            mask.save(mask_path)
            with self.assertRaisesRegex(ValueError, "not binary"):
                prepare_assets(source, root / "runtime", source_id="000000", selection_count=1)


if __name__ == "__main__":
    unittest.main()
