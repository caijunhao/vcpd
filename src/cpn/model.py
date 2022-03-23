from cpn.utils import interpolation as ipl
from torch import nn
import torch.nn.functional as f
import torch


class CPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv00 = ConvBlock3D(1, 16, stride=2, norm=True, relu=True)

        self.conv10 = ConvBlock3D(16, 32, stride=2, norm=True, relu=True)
        self.conv11 = ConvBlock3D(32, 32, stride=1, norm=True, relu=True)
        self.conv12 = ConvBlock3D(32, 32, stride=1, norm=True, relu=True)
        self.conv13 = ConvBlock3D(32, 32, stride=1, norm=True, relu=True)

        self.conv20 = ConvBlock3D(32, 64, stride=2, norm=True, relu=True)
        self.conv21 = ConvBlock3D(64, 64, stride=1, norm=True, relu=True)
        self.conv22 = ConvBlock3D(64, 64, stride=1, norm=True, relu=True)
        self.conv23 = ConvBlock3D(64, 64, stride=1, norm=True, relu=True)

        self.conv30 = ConvBlock3D(64, 128, stride=2, norm=True, relu=True)
        self.conv31 = ConvBlock3D(128, 256, stride=1, norm=True, relu=True)
        self.conv32 = ConvBlock3D(256, 256, stride=1, norm=True, relu=True)

        self.conv40 = ConvBlock3D(256, 64, norm=True, relu=True, upsm=True)
        self.conv41 = ConvBlock3D(64, 64, norm=True, relu=True)

        self.conv50 = ConvBlock3D(64+64, 32, norm=True, relu=True, upsm=True)
        self.conv51 = ConvBlock3D(32, 32, norm=True, relu=True)

        self.conv60 = ConvBlock3D(32+32, 16, norm=True, relu=True, upsm=True)
        self.conv61 = ConvBlock3D(16, 16, norm=True, relu=True)

        self.conv70 = ConvBlock3D(16+16, 8, norm=True, relu=True, upsm=True)
        self.conv71 = ConvBlock3D(8, 8, norm=False, relu=False)

        self.pnb1 = PointNetBlock([(64+32+16+8)*2, 128, 128], norm=True, relu=True)
        self.pnb2 = PointNetBlock([128, 128, 1], norm=False, relu=False)

    def forward(self, samples):
        x = samples['sdf_volume']
        # pcp: positive contact point; ncp: negative contact point
        ids_cp1, ids_cp2 = samples['ids_cp1'], samples['ids_cp2']
        x0 = self.conv00(x)

        x1 = self.conv10(x0)
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        x1 = self.conv13(x1)

        x2 = self.conv20(x1)
        x2 = self.conv21(x2)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)

        x3 = self.conv30(x2)
        x3 = self.conv31(x3)
        x3 = self.conv32(x3)

        x4 = self.conv40(x3)
        x4 = self.conv41(x4)

        x5 = self.conv50(torch.cat([x4, x2], dim=1))
        x5 = self.conv51(x5)

        x6 = self.conv60(torch.cat([x5, x1], dim=1))
        x6 = self.conv61(x6)

        x7 = self.conv70(torch.cat([x6, x0], dim=1))
        x7 = self.conv71(x7)

        features = torch.cat([ipl(x4, ids_cp1/8), ipl(x5, ids_cp1/4), ipl(x6, ids_cp1/2), ipl(x7, ids_cp1),
                              ipl(x4, ids_cp2/8), ipl(x5, ids_cp2/4), ipl(x6, ids_cp2/2), ipl(x7, ids_cp2)],
                             dim=1)
        x8 = self.pnb1(features)
        out = self.pnb2(x8)
        return out

    def load_network_state_dict(self, device, pth_file, strict=True):
        state_dict = torch.load(pth_file, device)
        self.load_state_dict(state_dict=state_dict, strict=strict)


class ConvBlock3D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3,
                 stride=1, norm=False, relu=False, pool=False, upsm=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=1, bias=not norm)
        self.norm = nn.GroupNorm(min(planes, 16), planes) if norm else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.relu is None else self.relu(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else f.interpolate(out, scale_factor=2, mode='trilinear', align_corners=True)

        return out


class ConvBlock2D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3,
                 stride=1, norm=False, relu=False, pool=False, upsm=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=1, bias=not norm)
        self.norm = nn.GroupNorm(min(planes, 16), planes) if norm else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if pool else None
        self.upsm = upsm

    def forward(self, x):
        out = self.conv(x)

        out = out if self.norm is None else self.norm(out)
        out = out if self.relu is None else self.relu(out)
        out = out if self.pool is None else self.pool(out)
        out = out if not self.upsm else f.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return out


class PointNetBlock(nn.Module):
    def __init__(self, channel_list, norm=True, relu=True):
        super().__init__()
        self.num_layer = len(channel_list) - 1
        for i in range(self.num_layer):
            relu_flag = relu if i == self.num_layer else True
            self.__setattr__('mlp{}'.format(i), ConvBlock2D(channel_list[i], channel_list[i+1],
                                                            kernel_size=1, stride=1,
                                                            norm=norm, relu=relu_flag))

    def forward(self, x):
        # The dimension of x should be B * C * N * 1
        for i in range(self.num_layer):
            x = self.__getattr__('mlp{}'.format(i))(x)
        return x

