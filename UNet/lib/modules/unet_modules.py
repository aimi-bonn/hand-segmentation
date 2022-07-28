import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_type=None, act_type=nn.PReLU,
                 padding=0, padding_mode="zeros"):
        super(UNetConvBlock, self).__init__()

        self.norm_type = norm_type
        layers = []

        if norm_type == nn.BatchNorm2d or norm_type == nn.InstanceNorm2d:
            args = args2 = (out_channels,)
        elif norm_type == nn.LayerNorm:
            # TODO: Replace dirty hack with actual nn.LayerNorm
            # C,H,W
            self.norm_type = nn.GroupNorm
            args = (1, out_channels)
        elif norm_type == nn.GroupNorm:
            # TODO: Don't hardcode the number of groups
            # Groups, Channels
            args = (8, out_channels)
        elif norm_type is None:
            args = ()
        else:
            print(f"Unknown norm_type: {norm_type}")
            raise NotImplementedError()

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode))
        layers.append(act_type())
        if norm_type is not None:
            layers.append(self.norm_type(*args))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode))
        layers.append(act_type())
        if norm_type is not None:
            layers.append(self.norm_type(*args))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        # print(f"feature map size in conv_block: {x.shape}, out conv_block: {y.shape}")
        return y


class UNetUpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_type=None, act_type=nn.PReLU,
                 padding=0, padding_mode="zeros", up_type="upconv", middle_channels=-1):
        super(UNetUpConvBlock, self).__init__()

        if middle_channels == -1:
            middle_channels = out_channels

        self.norm_type = norm_type
        if up_type == "upconv":
            self.up = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2)
        elif up_type == "upsample":
            self.up = nn.Sequential(nn.Upsample(mode="bilinear", scale_factor=2),
                                    nn.Conv2d(in_channels, middle_channels, kernel_size=1))
        else:
            print(f"Invalid up_type! Only supporting 'upconv' and 'upsample' ({up_type} given).")
            raise NotImplementedError()

        self.conv_block = UNetConvBlock(middle_channels * 2, out_channels, kernel_size=kernel_size, norm_type=norm_type,
                                    act_type=act_type, padding=padding, padding_mode=padding_mode)

    def forward(self, x, residual):
        ## Padding from Pytorch-UNet-master
        x = self.up(x)

        if type(residual) != type(None):
            # input is CHW
            # We check the diffX and diffY to ensure we're concatenating correctly, even if we had an odd size before
            diffY = residual.size()[2] - x.size()[2]
            diffX = residual.size()[3] - x.size()[3]

            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

            x = torch.cat([residual, x], dim=1)
            x = self.conv_block(x)

        return x
