import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modules.unet_modules import *

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=3, depth=5, norm_type=None, act_type=nn.PReLU,
                 padding=0, padding_mode="zeros", up_type="upsample"):
        super(UNet, self).__init__()

        self.depth = depth

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Encoder
        for i in range(depth-1):
            if i == 0:
                self.encoder_blocks.append(UNetConvBlock(in_channels, 2 ** (5+i), kernel_size=kernel_size,
                                                         norm_type=norm_type, act_type=act_type, padding=padding,
                                                         padding_mode=padding_mode))
                print(f"Encoder block #{i}, in_c: {in_channels}, out_c: {2 ** (5+i)}")
            else:
                self.encoder_blocks.append(UNetConvBlock(2 ** (4+i), 2 ** (5+i), kernel_size=kernel_size,
                                                         norm_type=norm_type, act_type=act_type, padding=padding,
                                                         padding_mode=padding_mode))
                print(f"Encoder block #{i}, in_c: {2 ** (4+i)}, out_c: {2 ** (5 + i)}")

        self.bottom_block = UNetConvBlock(2 ** (3+depth), 2 ** (4+depth), kernel_size=kernel_size,
                                          norm_type=norm_type, act_type=act_type, padding=padding,
                                          padding_mode=padding_mode)
        print(f"Bottom block, in_c: {2 ** (3+depth)}, out_c: {2 ** (4+depth)}")

        ## Decoder
        for j in range(depth-1, 0, -1):
            self.decoder_blocks.append(UNetUpConvBlock(2 ** (5+j), 2 ** (4+j), kernel_size=kernel_size,
                                                       norm_type=norm_type, act_type=act_type, padding=padding,
                                                       padding_mode=padding_mode, up_type=up_type))
            print(f"Decoder block #{j}, in_c: {2 ** (5+j)}, out_c: {2 ** (4+j)}")

        # Expected number of input channels = 64
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0)
        print(f"Out block, in_c: {32}, out_c: {num_classes}")

    def forward(self, x):
        residual_list = []

        ## Encoder
        for block in self.encoder_blocks:
            x = block(x)
            residual_list.append(x)
            x = self.pool(x)

        x = self.bottom_block(x)

        ## Decoder
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x, residual=residual_list[len(self.encoder_blocks)-(i+1)])

        x = self.out_conv(x)

        # Some mention we should use Sigmoid activation, others use no activation
        # return F.sigmoid(x)

        return x


def crop_residual(x_in, residual_in):
    # print(f"Size x: {x_in.shape}")
    # print(f"Size res_in: {residual_in.shape}")

    _,_, x_h, x_w = x_in.shape
    _,_, res_h, res_w = residual_in.shape

    assert res_h >= x_h, f"x height expected to be lower than or equal to residual height (got x_h: {x_h} and res_h: {res_h})"
    assert res_w >= x_w, f"x width expected to be lower than or equal to residual width (got x_w: {x_w} and res_w: {res_w})"

    diff_h = (res_h - x_h) // 2
    diff_w = (res_w - x_w) // 2

    cropped_residual = residual_in[:, :, diff_h:x_h+diff_h, diff_w:x_w+diff_w]
    # print(f"Size res_out: {cropped_residual.shape}")

    return cropped_residual






