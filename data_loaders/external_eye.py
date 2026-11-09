"""External Eye dataset used by the ViT training pipelines."""

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data_loaders.cnn import load_trainval_list
from data_loaders.labels import TASK_INDEX, binary_label, filter_valid_samples


GROUP_VALUES = [
    ["0", "1"],
    ["Asian", "Black or African American", "White", "Other or Unknown"],
    ["0", "1", "Unknown or Not Reported"],
    ["English", "Spanish", "Other or Unknown"],
]


class ExternalEyeDataset(Dataset):
    def __init__(self, args, train=True, transform=None, transform2=None, gen_attn=False):
        del gen_attn  # Retained for compatibility with existing training calls.
        self.samples = filter_valid_samples(load_trainval_list(args, train), args.task)

        if args.group_idx >= 0:
            group_type = int(args.group_type)
            group_value = GROUP_VALUES[group_type][args.group_idx]
            self.samples = [
                sample for sample in self.samples if sample[group_type + 2] == group_value
            ]

        self.image_root = "data/images"
        self.task_name = args.task
        self.task_index = TASK_INDEX[args.task]
        self.transform = transform
        self.transform2 = transform2
        self.labels = np.asarray(
            [binary_label(sample[self.task_index], self.task_name) for sample in self.samples]
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        grade = sample[0]
        image_name = sample[1]
        sex, race, ethnic, language = sample[2:6]

        label = binary_label(sample[self.task_index], self.task_name)

        image = Image.open(os.path.join(self.image_root, image_name)).convert("RGB")
        sample_name = f"{image_name}_{grade}"

        if self.transform and self.transform2 is None:
            return [
                self.transform(image),
                255,
                sample_name,
                sex,
                race,
                ethnic,
                language,
                label,
            ]

        if self.transform and self.transform2:
            return [
                [self.transform(image), self.transform2(image)],
                255,
                os.path.join(self.image_root, image_name),
                sex,
                race,
                ethnic,
                language,
                label,
            ]

        return [image, 255, sample_name, sex, race, ethnic, language, label]

    def __len__(self):
        return len(self.samples)
