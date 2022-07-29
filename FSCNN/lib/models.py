import logging
import os.path
from typing import Dict, List, Union

import pytorch_lightning as pl
import numpy as np
import torch
import torchmetrics
from scipy.special import softmax
from time import time
from lib.modules.losses import CombinedLoss

from lib.models import *
import matplotlib.pyplot as plt

from torch import nn
from lib import sub_module as sm


class MaskModel(pl.LightningModule):
    def __init__(
        self, kernel_size: tuple = (5, 5), num_filters: int = 32,
    ):
        super(MaskModel, self).__init__()
        self.backbone = FastSurferCNN(
            n_classes=2,
            n_input_channels=1,
            kernel_size=kernel_size,
            num_filters=num_filters,
        )
        self.loss = CombinedLoss()
        self.dice = DiceEvaluator(2, return_iou=True)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        _, d = self._shared_step(batch)
        self.log_dict(
            {"train_" + k: v for k, v in d.items()}, on_step=False, on_epoch=True
        )
        return d

    def validation_step(self, batch, batch_idx):
        _, d = self._shared_step(batch)
        self.log_dict(
            {"val_" + k: v for k, v in d.items()}, on_step=False, on_epoch=True
        )
        return d

    def _shared_step(self, batch):
        x = batch["image"]
        y = batch["mask"]
        w = batch["weight"]

        y_hat = self.forward(x)
        loss, ce_loss, dice_loss = self.loss(y_hat, y, w)

        dice, _, _, iou = self.dice(y_hat, y)
        summary_dict = {
            "loss": loss,
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
            "1-dice": 1 - dice,
            "1-iou": 1 - iou,
        }
        return y_hat, summary_dict


class FastSurferCNN(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)

    NOTE: the models do not compute soft-maxed values since it is calculated intrinsically inside of the loss

    from https://github.com/sRassmann/mm-Fat-Seg/blob/master/lib/models.py commit 8b6b1a0
    """

    defaults = {
        "num_channels": 1,
        "num_filters": 2,  # usually 64
        "stride_conv": 1,
        "pool": 2,
        "stride_pool": 2,
        "kernel_c": 1,
        "kernel_d": 1,
        "height": 256,
        "width": 256,
    }

    def __init__(
        self,
        n_classes: int = 5,
        n_input_channels: int = 4,
        kernel_size: tuple = (5, 5),
        num_filters: int = 2,
    ):
        """
        :param n_classes: number of classes in the output final layer
        :param kernel_size: size of the kernel as (height, width)
        :param num_filters: number of filter for each Conv layer
        """

        params = self.defaults
        params["batch_size"] = 16
        params["num_classes"] = n_classes
        params["kernel_h"] = kernel_size[0]
        params["kernel_w"] = kernel_size[1]
        params["num_channels"] = n_input_channels
        params["num_filters"] = num_filters

        super(FastSurferCNN, self).__init__()

        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params["num_channels"] = params["num_filters"]
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params["num_channels"] = params["num_filters"]
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        params["num_channels"] = params["num_filters"]
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(
            encoder_output1
        )
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(
            encoder_output2
        )
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(
            encoder_output3
        )

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(
            decoder_output4, skip_encoder_3, indices_3
        )
        decoder_output2 = self.decode2.forward(
            decoder_output3, skip_encoder_2, indices_2
        )
        decoder_output1 = self.decode1.forward(
            decoder_output2, skip_encoder_1, indices_1
        )

        logits = self.classifier.forward(decoder_output1)

        return logits


class DiceEvaluator:
    def __init__(self, num_classes: int = 5, return_iou: bool = False):
        self.num_classes = num_classes
        self.eps = 1e-5
        self.return_iou = return_iou

    def __call__(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        mask: Union[torch.Tensor, float] = 1.0,
    ):
        """
        Calculate average and class-wise average DICE score as well as the dice confusion matrix
        :param y: Ground truth Tensor (N x H X W) or (N x 1 x H X W) whereby the entries encode the labels
        :param y_hat: One-hot encoded softmax tensor (N X C x H x W), whereby C denotes the number of classes)
        :param mask: masks pixel to be neglected for calculating the score
        :return:
        """
        _, preds = torch.max(y_hat.detach(), dim=1)
        if len(y.shape) == 4:
            y = y.squeeze(dim=1)
        if isinstance(mask, torch.Tensor) and len(mask.shape) == 4:
            mask = mask.squeeze(dim=1)

        dice_cm = torch.zeros(self.num_classes, self.num_classes, y.shape[0])
        ious = []
        for i in range(self.num_classes):
            gt = (y == i) * mask

            for j in range(self.num_classes):
                pred = (preds == j) * mask
                inter = torch.sum(gt * pred, axis=(1, 2))
                union = torch.sum(gt, axis=(1, 2)) + torch.sum(pred, axis=(1, 2))
                union = union + self.eps

                dice_cm[i, j] = (2 * inter + self.eps) / union

                if i == j:
                    iou = (inter + self.eps) / (
                        torch.sum((pred + gt) > 0, axis=(1, 2)) + self.eps
                    )
                    ious.append(torch.mean(iou))

        avg_dice = torch.mean(dice_cm, axis=2)
        class_dices = torch.diagonal(avg_dice)

        if not self.return_iou:
            return torch.mean(class_dices), class_dices, avg_dice
        else:
            return (
                torch.mean(class_dices),
                class_dices,
                avg_dice,
                torch.mean(torch.tensor(ious)),
            )

    def iou_score(self, y_hat, y):
        """
        compute the intersection-over-union score
        both inputs should be categorical (as opposed to one-hot)
        """
        intersect_ = []
        union_ = []

        for i in range(1, self.num_classes):
            intersect = ((y_hat == i).float() + (y == i).float()).eq(
                2
            ).sum().item() + self.eps
            union = ((y_hat == i).float() + (y == i).float()).ge(
                1
            ).sum().item() + self.eps
            intersect_.append(intersect)
            union_.append(union)

        return np.mean(np.array(intersect_) / np.array(union_))
