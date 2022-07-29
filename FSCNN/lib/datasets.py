"""
Module to manage datasets and loaders
"""
import os
import re
from functools import cache
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import yaml
from glob import glob

import random
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from torchvision import transforms

from typing import Union, List

import logging

logger = logging.getLogger(__name__)


class MaskDataSet(Dataset):
    def __init__(
        self, path="data/train", aug=None, weight=5,
    ):
        self.paths = glob(path + "/" + ("[0-9]" * 4) + ".png")
        self.aug = aug
        self._init_weights(weight)

    def _init_weights(self, weight):
        self.masks = []
        for path in self.paths:
            if not os.path.exists(path.replace(".png", "_weights.png")):
                rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.uint8)

                img = rgba[:, :, 0]
                mask = rgba[:, :, 3] // 255

                thr = np.quantile(img, 0.8)
                w = 1 + weight * (img > thr) * (1 - mask) + mask

                dx, dy = np.gradient(mask)
                w = (
                    w
                    + cv2.blur(((dx ** 2 + dy ** 2) > 0).astype(np.uint8), (20, 20))
                    * 5
                    * weight
                )
                w = (w / 200) * (2 ** 16 - 1)
                cv2.imwrite(path.replace(".png", "_weights.png"), w.astype(np.uint16))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rgba = cv2.imread(self.paths[i], cv2.IMREAD_UNCHANGED).astype(np.uint8)

        img = rgba[:, :, :3]
        mask = rgba[:, :, 3]
        w = cv2.imread(
            self.paths[i].replace(".png", "_weights.png"), cv2.IMREAD_UNCHANGED
        )
        w = w / (2 ** 16 - 1) * 100

        if np.random.rand() < 0.01:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.where(mask, np.quantile(img, 0.05).astype(np.uint8), img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.blur(img, (7, 7))
            mask = np.zeros_like(img[:, :, 0])

        img = self.specific_aug(img)
        augment = self.aug(image=img, mask=mask, weight=w)
        img = self.normalize(augment["image"][0].unsqueeze(dim=0))

        if random.random() < 0.075:
            img = img * 0.1

        mask = torch.div(augment["mask"], 255, rounding_mode="trunc")
        w = augment["weight"]

        return {"image": img, "mask": mask, "weight": w}

    @staticmethod
    def normalize(img: torch.Tensor) -> torch.Tensor:
        img -= img.min()
        return img / img.max()

    @staticmethod
    def specific_aug(img):
        """
        augmentations specific for the given scans. credits to Alexander Hustinx
        """
        img = Image.fromarray(img).convert("RGBA")  # stuff is in PIL
        img = MaskDataSet.alex_aug(img)
        return np.asarray(img)

    @staticmethod
    def alex_aug(img):
        """
        uses PIL images
        """
        w, h = img.size
        # Add color gradient edge to image
        if random.random() > 0.5:
            line_thickness = random.random() * 0.03  # Max 5% of image width or height

            def interpolate(f_co, t_co, interval):
                det_co = [(t - f) / interval for f, t in zip(f_co, t_co)]
                for i in range(interval):
                    yield [round(f + det * i) for f, det in zip(f_co, det_co)]

            gradient = Image.new("RGBA", img.size, color=0)
            img_draw = ImageDraw.Draw(gradient)

            # Horizontal edge
            if random.random() > 0.5:
                from_top = True if random.random() > 0.5 else False
                # line_thickness = int(line_thickness * h + 0.02 * h)
                for i, color in enumerate(
                    interpolate(
                        (255, 255, 255), (0, 0, 0), int(line_thickness * h + 0.02 * h)
                    )
                ):
                    color.append(127)
                    img_draw.line(
                        ((0, i if from_top else h - i), (w, i if from_top else h - i)),
                        tuple(color),
                        width=1,
                    )

            # Vertical edge
            if random.random() > 0.5:
                from_left = True if random.random() > 0.5 else False
                # line_thickness = int(line_thickness * w + 0.02 * w)
                for i, color in enumerate(
                    interpolate(
                        (255, 255, 255), (0, 0, 0), int(line_thickness * w + 0.02 * w)
                    )
                ):
                    color.append(127)
                    img_draw.line(
                        (
                            (i if from_left else w - i, 0),
                            (i if from_left else w - i, h),
                        ),
                        tuple(color),
                        width=1,
                    )

            img = Image.alpha_composite(img, gradient)

            # Add a random gray patch to the image
            intensity = random.random()
            if intensity > 0.5:
                rect = Image.new("RGBA", img.size)
                img_draw = ImageDraw.Draw(rect, "RGBA")
                x, y = random.random() * w, random.random() * h
                x2, y2 = random.random() * w, random.random() * h
                intensity = int(200 * intensity)
                img_draw.rectangle(
                    ((x, y), (x2, y2)), fill=(intensity, intensity, intensity, 127)
                )
                img = Image.alpha_composite(img, rect)
                # plt.imshow(img)
                # plt.show()

            # Overlay the entire image with a high intensity rectangle
            intensity = random.random()
            if intensity > 0.80:
                rect = Image.new("RGBA", img.size)
                img_draw = ImageDraw.Draw(rect, "RGBA")
                x, y = 0, 0
                x2, y2 = img.size
                intensity = int(200 * intensity)
                img_draw.rectangle(
                    ((x, y), (x2, y2)), fill=(intensity, intensity, intensity, 127)
                )
                img = Image.alpha_composite(img, rect)
            # Adjust image gamma to brighten aspects of the image
            elif random.random() > 0.75:
                gamma = random.random() * 0.5 + 0.4  # in range [0.4, 0.9]
                img = transforms.functional.adjust_gamma(img, gamma)
        return img.convert("RGB")


class PretrainDataset(MaskDataSet):
    def __init__(
        self,
        img_path="../../data/annotated/rsna_bone_age/bone_age_training_data_set",
        mask_path="../../data/masks/tensormask/bone_age",
        weight_path="../../data/masks/weights/",
        aug=None,
        weight=5,
    ):
        """uses existing masks (e.g. Tensormask). Intended for pretraining"""
        paths = glob(img_path + "/*.png")
        self.paths = []
        self.masks = []
        self.weights = []
        for path in paths:
            name = os.path.basename(path)
            if os.path.exists(os.path.join(mask_path, name)):
                self.paths.append(path)
                self.masks.append(os.path.join(mask_path, name))
        self.aug = aug
        self._init_weights(weight, weight_path)

    def _init_weights(self, weight, weight_path="../../data/masks/weights/"):
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
            for i, path in tqdm(enumerate(self.paths)):
                name = os.path.basename(path)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                mask = (
                    cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    // 255
                )

                thr = np.quantile(img, 0.8)
                w = 1 + weight * (img > thr) * (1 - mask) + mask

                dx, dy = np.gradient(mask)
                w = (
                    w
                    + cv2.blur(
                        ((dx ** 2 + dy ** 2) > 0).astype(np.uint8) * 255, (20, 20)
                    )
                    / 255
                    * 5
                    * weight
                )
                w = (w / 200) * (2 ** 16 - 1)
                cv2.imwrite(os.path.join(weight_path, name), w.astype(np.uint16))
        self.weights = glob(os.path.join(weight_path, "*"))

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        w = cv2.imread(self.weights[i], cv2.IMREAD_UNCHANGED)

        w = w / (2 ** 16 - 1) * 100

        if np.random.rand() < 0.01:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.where(mask, np.quantile(img, 0.05).astype(np.uint8), img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.blur(img, (7, 7))
            mask = np.zeros_like(img[:, :, 0])

        img = self.specific_aug(img)
        if w.shape != mask.shape:
            print(self.masks[i])
            w = np.ones_like(mask)
        augment = self.aug(image=img, mask=mask, weight=w)
        img = self.normalize(augment["image"][0].unsqueeze(dim=0))

        if random.random() < 0.075:
            img = img * 0.1

        mask = torch.div(augment["mask"], 255, rounding_mode="trunc")
        w = augment["weight"]
        return {"image": img, "mask": mask, "weight": w}


class MaskModule(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size=8,
        test_batch_size=8,
        num_workers=8,
        train_path="data/train",
        val_path="data/val",
        weight_path="../data/masks/weights/",
        size=512,
        pretrain=False,
    ):
        super(MaskModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        if pretrain:
            self.train = PretrainDataset(
                aug=self.get_default_train_aug(size), weight_path=weight_path,
            )
        else:
            self.train = MaskDataSet(train_path, aug=self.get_default_train_aug(size))
        self.val = MaskDataSet(val_path, aug=self.get_inference_aug(size))

        logger.info(f"before {len(self.train)}")
        val = [os.path.basename(x) for x in self.val.paths]
        self.train.paths = [
            x for x in self.train.paths if not os.path.basename(x) in val
        ]
        self.train.masks = [
            x for x in self.train.masks if not os.path.basename(x) in val
        ]
        if isinstance(self.train, PretrainDataset):
            self.train.weights = [
                x for x in self.train.weights if not os.path.basename(x) in val
            ]
        logger.info(f"after {len(self.train)}")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.test_batch_size, num_workers=self.num_workers,
        )

    @staticmethod
    def get_inference_aug(size=512):
        return A.Compose(
            [
                A.RandomResizedCrop(size, size, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                ToTensorV2(),
            ],
            additional_targets={"weight": "mask"},
        )

    @staticmethod
    def get_default_train_aug(size=512):
        return A.Compose(
            [
                A.Sharpen(p=0.5, alpha=0.2),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.CLAHE(p=0.2),
                        A.RandomBrightnessContrast(
                            p=0.6,
                            brightness_limit=(0.3, 0.6),
                            contrast_limit=(-0.9, -0.7),
                            brightness_by_max=True,
                        ),
                        A.RandomGamma((50, 200), p=0.2),
                    ],
                    p=0.95,
                ),
                A.OneOf(
                    [
                        A.Affine(
                            scale=(0.50, 1.25),
                            rotate=(-45, 45),
                            shear=(-5, 5),
                            mode=1,
                            p=0.5,
                        ),
                        A.Affine(
                            scale=(0.50, 1.25),
                            rotate=(-45, 45),
                            shear=(-5, 5),
                            mode=cv2.BORDER_CONSTANT,
                            cval=245,
                            p=0.5
                            # simulate SHOX scan frame
                        ),
                    ]
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=1, sigma=2 * 0.05, alpha_affine=10 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT),
                        A.OpticalDistortion(
                            distort_limit=0.1,
                            shift_limit=0.2,
                            p=0.5,
                            border_mode=cv2.BORDER_CONSTANT,
                        ),
                    ],
                    p=0.5,
                ),
                A.RandomResizedCrop(size, size, scale=(1.0, 1.0), ratio=(0.8, 1.25)),
                A.OneOf(
                    [
                        A.augmentations.transforms.GaussNoise(var_limit=(10, 40)),
                        A.augmentations.transforms.ImageCompression(
                            quality_lower=25, quality_upper=80
                        ),
                    ],
                    p=0.5,
                ),
                A.InvertImg(p=0.3),
                ToTensorV2(),
            ],
            additional_targets={"weight": "mask"},
        )
