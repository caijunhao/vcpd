from vpn.utils import interpolation_v2, extract_sdf
from torch import nn
import torch.nn.functional as f
import torch


class VPN(nn.Module):
    def __init__(self, output_channel=18):
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
        self.conv31 = ConvBlock3D(128, 128, stride=1, norm=True, relu=True)
        self.conv32 = ConvBlock3D(128, 128, stride=1, norm=True, relu=True)
        # self.conv31 = ResBlock3D(128, 128, expansion=1)
        # self.conv32 = ResBlock3D(128, 128, expansion=4)

        self.conv40 = ConvBlock3D(128, 64, norm=True, relu=True, upsm=True)
        self.conv41 = ConvBlock3D(64, 64, norm=True, relu=True)

        self.conv50 = ConvBlock3D(64+64, 32, norm=True, relu=True, upsm=True)
        self.conv51 = ConvBlock3D(32, 32, norm=True, relu=True)

        self.conv60 = ConvBlock3D(32+32, 16, norm=True, relu=True, upsm=True)
        self.conv61 = ConvBlock3D(16, 16, norm=True, relu=True)

        self.conv70 = ConvBlock3D(16+16, 8, norm=True, relu=True, upsm=True)
        self.conv71 = ConvBlock3D(8, 8, norm=False, relu=False)

        # self.pnb = PointNetBlock([128+64+32+16+8, 128, output_channel], norm=False, relu=False)
        self.pnb = PointNetBlock([3+64+32+16+8, 128, output_channel], norm=False, relu=False)

    def forward(self, samples):
        x = samples['sdf_volume']
        ids = samples['ids']
        normals = samples['normals']
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

        f4 = interpolation_v2(x4, ids / 8)
        f5 = interpolation_v2(x5, ids / 4)
        f6 = interpolation_v2(x6, ids / 2)
        f7 = interpolation_v2(x7, ids)
        features = torch.cat([normals, f4, f5, f6, f7], dim=1)
        out = self.pnb(features)
        return out

    def load_network_state_dict(self, device, pth_file, strict=True):
        state_dict = torch.load(pth_file, device)
        self.load_state_dict(state_dict=state_dict, strict=strict)


class RefineNetV0(nn.Module):
    def __init__(self, num_sample):
        super().__init__()
        self.conv00 = ConvBlock3D(4, 16, stride=1, norm=True, relu=True)
        self.conv01 = ConvBlock3D(16, 16, stride=2, norm=True, relu=True)
        self.conv10 = ConvBlock3D(16, 32, stride=1, norm=True, relu=True)
        self.conv11 = ConvBlock3D(32, 32, stride=2, norm=True, relu=True)
        self.conv20 = ConvBlock3D(32, 64, stride=1, norm=True, relu=True)
        self.conv21 = ConvBlock3D(64, 64, stride=2, norm=True, relu=True)
        self.conv30 = ConvBlock3D(64, 128, stride=1, norm=True, relu=True)
        self.conv31 = ConvBlock3D(128, 256, stride=2, norm=True, relu=True)
        self.conv40 = ConvBlock3D(256, 512, stride=1, norm=False, relu=False)
        self.conv41 = ConvBlock3D(512, 6, stride=1, norm=False, relu=False)
        self.num_sample = num_sample

    def forward(self, samples):
        sdf = samples['sdf_volume']
        pos = samples['perturbed_pos']
        origin = samples['origin']
        resolution = samples['resolution']
        sdf_vec = extract_sdf(pos, sdf, origin, resolution)  # batch * 1 * num_sdf * num_pose
        batch, _, num_sdf, num_pose = sdf_vec.shape
        assert num_pose == 1
        sdf_vol = torch.squeeze(torch.squeeze(sdf_vec, dim=-1), dim=1)  # batch * num_sdf
        assert num_sdf == self.num_sample ** 3 * 4
        # sdf_vol = (sdf_vol <= 0).to(torch.float32)
        sdf_vol = sdf_vol.view(batch, self.num_sample ** 3, 4)
        sdf_vol = sdf_vol.view(batch, self.num_sample, self.num_sample, self.num_sample, 4)
        sdf_vol = sdf_vol.permute(0, 4, 1, 2, 3)  # batch * 4 * num_sample * num_sample * num_sample
        sdf_vol = self.conv00(sdf_vol)
        sdf_vol = self.conv01(sdf_vol)
        sdf_vol = self.conv10(sdf_vol)
        sdf_vol = self.conv11(sdf_vol)
        sdf_vol = self.conv20(sdf_vol)
        sdf_vol = self.conv21(sdf_vol)
        sdf_vol = self.conv30(sdf_vol)
        sdf_vol = self.conv31(sdf_vol)
        sdf_vol = self.conv40(sdf_vol)
        sdf_vol = self.conv41(sdf_vol)
        out = torch.squeeze(sdf_vol, dim=-1)  # batch * 9 * 1 * 1

        a1, a2 = out[:, 0:3], out[:, 3:6]
        b1 = a1 / torch.linalg.norm(a1, dim=1, keepdim=True)
        b2 = a2 - torch.sum(a2 * b1, dim=1, keepdim=True) * b1
        b2 = b2 / torch.linalg.norm(b2, dim=1, keepdim=True)
        b3 = torch.cross(b1, b2, dim=1)
        delta_rot = torch.cat([b1, b2, b3], dim=2)  # B * 3 * 3 * num_pose
        return delta_rot

    def load_network_state_dict(self, device, pth_file, strict=True):
        state_dict = torch.load(pth_file, device)
        self.load_state_dict(state_dict=state_dict, strict=strict)


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


class ResBlock3D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, expansion=1):
        super().__init__()
        bottleneck = max(inplanes // expansion, 1)
        if inplanes != planes or stride != 1:
            self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv1 = nn.Conv3d(inplanes, bottleneck, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.GroupNorm(min(bottleneck, 16), bottleneck)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(bottleneck, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.GroupNorm(min(planes, 16), planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'conv0'):
            residual = self.conv0(x)

        out += residual
        out = self.relu(out)

        return out

