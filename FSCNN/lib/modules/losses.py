# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
from typing import *

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

import matplotlib.pyplot as plt


class DiceLoss(_Loss):
    """
    2D/3D Dice Loss
    DOES NOT INCLUDE SOFTMAX AS A PART. MUST MANUALLY PROVIDE SOFT INPUTS
    """

    def forward(
        self, y_hat, y, weights=1, channel_weights=None, ignore_index=None, eps=1e-3
    ):
        """
        :param y_hat: N x C x H x W or N x C x H x W x D Variable
        :param y: N x 1 x C x W or N x C x W
        :param weights: C FloatTensor with class wise weights
        :param int ignore_index: ignore label with index x in the loss calculation
        :return:
        """
        # create zeros array with input size N x C x H x W
        encoded_target = torch.zeros_like(y_hat)
        if len(y.shape) == 3:  # add axis if missing
            y = y.unsqueeze(1)

        if ignore_index is not None:
            mask = y == ignore_index
            y = y.clone()
            y[mask] = 0
            encoded_target.scatter_(1, y, 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0

        else:
            # One hot encoding
            encoded_target.scatter_(1, y, 1)

        if channel_weights is None:
            channel_weights = 1

        intersection = y_hat * encoded_target * weights
        denominator = y_hat * weights + encoded_target * weights
        if ignore_index is not None:
            denominator[mask] = 0

        # same as adding it to each intersection entry but likely faster
        eps = eps * y.shape[0] * y.shape[2] * y.shape[3]

        # 2D N X C X H X W
        if len(y_hat.shape) == 4:
            intersection = intersection.sum((0, 2, 3))
            denominator = denominator.sum((0, 2, 3))

        # 3D N X C x D X H X W
        elif len(y_hat.shape) == 5:
            intersection = intersection.sum((0, 2, 3, 4))
            denominator = denominator.sum((0, 2, 3, 4))
            eps *= y.shape[4]

        loss_per_channel = channel_weights * (
            1 - ((2 * intersection + eps) / (denominator + eps))
        )  # Channel-wise weights
        return torch.mean(loss_per_channel)


class FocalCrossEntropy(nn.Module):
    """
    Focal loss implementation of the loss in Focal loss for Dense Object detection
        Args:
        -- inputx N x C x H x W
        -- target - N x H x W - int type
        -- weight - N x H x W - float # alpha
        -- gamma : weight penalization of
    """

    def __init__(self, weight=None, reduction="none", gamma=2):
        super(FocalCrossEntropy, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, reduction=reduction)
        self.gamma = gamma

    def forward(self, y_hat, y, weights=None):

        fl = torch.mul(
            torch.pow((1 - F.softmax(y_hat, 1)), self.gamma), F.log_softmax(y_hat, 1)
        )
        nll = self.nll_loss(fl, y)
        if not torch.is_tensor(weights):
            return nll
        else:
            return torch.mean(torch.mul(nll, weights))


class CrossEntropy(nn.Module):
    """
    Cross-entropy loss implemented as negative log likelihood
    """

    def __init__(self, weight=None, reduction="none"):
        super(CrossEntropy, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, y_hat, y, weights=None):
        nll = self.nll_loss(y_hat, y if len(y.shape) == 3 else y.squeeze(dim=1))
        if not torch.is_tensor(weights):
            return torch.mean(nll)
        else:
            return torch.mean(torch.mul(nll, weights))


class CombinedLoss(nn.Module):
    """
    For CrossEntropy the input has to be a long tensor
    Note: Only the combined loss is considered be relevant for backprop, the two summands (CE and DICE) are detached from the computational graph!
    Args:
        -- inputx N x C x H x W
        -- target - N x H x W - int type
        -- weight - N x H x W - float
    """

    def __init__(self, weight_dice=1, weight_ce=1):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropy()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, y_hat, y, weight):
        y = y.type(torch.LongTensor)  # Typecast to long tensor
        if y_hat.is_cuda:
            y = y.cuda()

        input_soft = F.softmax(y_hat, dim=1)  # Along Class Dimension
        dice_val = self.dice_loss(input_soft, y)
        ce_val = self.cross_entropy_loss.forward(y_hat, y, weight)
        total_loss = dice_val * self.weight_dice + ce_val * self.weight_ce

        return total_loss, dice_val.detach(), ce_val.detach()
