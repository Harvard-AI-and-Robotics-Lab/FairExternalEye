"""External Eye dataset used by the CNN training pipelines."""

import os
import random

import numpy as np
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from data_loaders.labels import TASK_INDEX, binary_label, filter_valid_samples


def load_trainval_list(args, train):
    del args  # Retained for compatibility with existing training calls.
    split = "train_final.npy" if train else "val_final.npy"
    return np.load(os.path.join("data", "splits", split), allow_pickle=True)


class ExternalEyeCNNDataset(Dataset):
    def __init__(self, args, resize, color_jitter, train):
        self.samples = filter_valid_samples(load_trainval_list(args, train), args.task)

        self.image_root = "data/images"
        self.resize = resize
        self.color_jitter = color_jitter
        self.train = train
        self.task_name = args.task
        self.task_index = TASK_INDEX[args.task]
        self.labels = np.asarray(
            [binary_label(sample[self.task_index], self.task_name) for sample in self.samples]
        )
        self._build_transforms()

    def _build_transforms(self):
        self.train_resize = transforms.RandomResizedCrop(
            size=int(self.resize),
            scale=(0.9, 1.1),
            ratio=(1, 1),
            interpolation=InterpolationMode.BICUBIC,
        )
        self.eval_resize = transforms.Resize(
            size=(int(self.resize), int(self.resize)),
            interpolation=InterpolationMode.BICUBIC,
        )
        self.jitter = transforms.RandomApply(
            [transforms.ColorJitter(*self.color_jitter)],
            p=0.8,
        )
        self.final_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

    def _transform(self, image):
        if self.train:
            image = self.train_resize(image)
            if random.random() > 0.5:
                image = TF.hflip(image)
            if self.task_name in {"dr", "amd"}:
                image = self.jitter(image)
        else:
            image = self.eval_resize(image)
        return self.final_transform(image)

    def __getitem__(self, index):
        sample = self.samples[index]
        grade = sample[0]
        image_name = sample[1]
        sex, race, ethnic, language = sample[2:6]

        label = binary_label(sample[self.task_index], self.task_name)

        image = Image.open(os.path.join(self.image_root, image_name)).convert("RGB")
        return [
            self._transform(image),
            255,
            f"{image_name}_{grade}",
            sex,
            race,
            ethnic,
            language,
            label,
        ]

    def __len__(self):
        return len(self.samples)
