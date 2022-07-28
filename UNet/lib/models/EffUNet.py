import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import Swish, MemoryEfficientSwish

from lib.models.EfficientNet import VALID_MODELS, EfficientNet
from lib.modules.unet_modules import *


class EffUNet(nn.Module):
    def __init__(self, in_channels, up_type='upconv', norm_type=None, act_type=MemoryEfficientSwish,
                 base='efficientnet-b0', pretrained=True):
        super(EffUNet, self).__init__()

        assert base in VALID_MODELS, f"Given base model type ({base}) is invalid"
        if pretrained:
            assert base != 'efficientnet-l2', "'efficientnet-l2' does not currently have pretrained weights"
            self.base = EfficientNet.from_pretrained(base, in_channels=in_channels)
        else:
            self.base = EfficientNet.from_name(base, in_channels=in_channels)

        ## Freeze base-EfficientNet model weights
        # for param in self.base.parameters():
        #     param.requires_grad = False

        self.act = act_type()

        ## Decoder
        self.decoder_blocks = nn.ModuleList()
        channel_sizes_out = [112, 40, 24, 16, 8]  # size of output = size of next residual
        channel_sizes_in = [1280, 112, 40, 24, 16]  # size of input = size of previous output

        for idx in range(len(channel_sizes_in)):
            self.decoder_blocks.append(UNetUpConvBlock(channel_sizes_in[idx], channel_sizes_out[idx],
                                                   kernel_size=3, norm_type=norm_type, act_type=act_type, padding=1,
                                                   padding_mode='zeros', up_type=up_type))
            print(f"Decoder block #{idx}, in_c: {channel_sizes_in[idx]}, out_c: {channel_sizes_out[idx]}")

        # Expected number of input channels = 8
        self.out_conv = nn.Conv2d(channel_sizes_out[-1], 1, kernel_size=1, stride=1, padding=0)
        print(f"Out block, in_c: {channel_sizes_out[-1]}, out_c: {1}")

    def forward(self, x):
        ## Encoder EfficientNet
        x, residual_list = self.base.extract_features(x, return_residual=True)

        ## Decoder
        # most recent is not being concatenated
        residual_list.pop()

        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x, residual=residual_list.pop() if i != (len(self.decoder_blocks) - 1) else None)

        x = self.out_conv(x)

        return x
