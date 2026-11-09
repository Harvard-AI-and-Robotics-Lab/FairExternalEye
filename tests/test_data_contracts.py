import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from data_loaders.cnn import ExternalEyeCNNDataset
from data_loaders.external_eye import ExternalEyeDataset
from data_loaders.labels import binary_label, is_valid_label
from tools.utils import build_dataset_trm


class DataContractTests(unittest.TestCase):
    def test_shared_label_definitions(self):
        cases = {
            "dr": {0: 0, 1: 1},
            "amd": {0: 0, 1: 0, 2: 0, 3: 1, 255: 0},
            "glaucoma": {-1: 0, 0: 0, 1: 1},
        }
        for task, labels in cases.items():
            for raw, expected in labels.items():
                self.assertTrue(is_valid_label(str(float(raw)), task))
                self.assertEqual(binary_label(str(float(raw)), task), expected)
        self.assertFalse(is_valid_label("255", "glaucoma"))

    def test_cnn_and_vit_loaders_use_identical_labels(self):
        rows = np.asarray(
            [
                ["g1", "a.jpg", "0", "Asian", "0", "English", "1", "3", "1"],
                ["g2", "b.jpg", "1", "White", "1", "Spanish", "0", "255", "255"],
            ],
            dtype=str,
        )
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                (root / "data" / "splits").mkdir(parents=True)
                (root / "data" / "images").mkdir()
                np.save(root / "data" / "splits" / "train_final.npy", rows)
                np.save(root / "data" / "splits" / "val_final.npy", rows)
                for image_name in ("a.jpg", "b.jpg"):
                    Image.new("RGB", (16, 16)).save(root / "data" / "images" / image_name)
                os.chdir(root)

                for task, expected_length in (("dr", 2), ("amd", 2), ("glaucoma", 1)):
                    args = SimpleNamespace(task=task, group_idx=-1, group_type=None)
                    cnn_dataset = ExternalEyeCNNDataset(args, 16, [0, 0, 0, 0], False)
                    vit_dataset = ExternalEyeDataset(args, False)
                    self.assertEqual(len(cnn_dataset), expected_length)
                    self.assertEqual(len(vit_dataset), expected_length)
                    self.assertEqual(cnn_dataset[0][-1], vit_dataset[0][-1])
                    self.assertEqual(cnn_dataset.labels.tolist(), vit_dataset.labels.tolist())

                transform_args = SimpleNamespace(
                    data_set="exEYE",
                    task="dr",
                    model="ViT_fscl",
                    group_idx=-1,
                    group_type=1,
                    input_size=16,
                    gen_attention_maps=False,
                    aa=None,
                    color_jitter=0.4,
                    center_crop=False,
                    scale=(0.8, 1.0),
                    train_interpolation="bicubic",
                    reprob=0.0,
                    remode="pixel",
                    recount=1,
                )
                fscl_dataset = build_dataset_trm(True, transform_args)
                self.assertEqual(len(fscl_dataset[0][0]), 2)
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
