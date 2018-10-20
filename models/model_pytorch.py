import numpy as np

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class cSELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(cSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class sSELayer(nn.Module):
    def __init__(self, channel):
        super(sSELayer, self).__init__()
        self.fc = nn.Conv2d(channel, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        y = self.sigmoid(y)
        return x * y


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 stride=(1, 1), padding=(1, 1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)), inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv3_bn(self.conv3(x)), inplace=True)
        return x


class DecoderV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)), inplace=True)

        residual1 = x
        x = F.relu(self.conv2_bn(self.conv2(x)), inplace=True)
        x = F.relu(self.conv3_bn(self.conv3(x)), inplace=True)
        x += residual1

        residual2 = x
        x = F.relu(self.conv4_bn(self.conv4(x)), inplace=True)
        x = F.relu(self.conv5_bn(self.conv5(x)), inplace=True)
        x += residual2

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        return x


class DecoderV3(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(DecoderV3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)

        self.channel_gate = cSELayer(out_channels)
        self.spatial_gate = sSELayer(out_channels)

    def forward(self, x, e):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = torch.cat([x, e], 1)

        x = F.relu(self.conv1_bn(self.conv1(x)), inplace=True)
        x = F.relu(self.conv2_bn(self.conv2(x)), inplace=True)

        g1 = self.channel_gate(x)
        g2 = self.spatial_gate(x)
        x = g1 + g2
        return x


class DecoderV4(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(DecoderV4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)

        self.channel_gate = cSELayer(out_channels)
        self.spatial_gate = sSELayer(out_channels)

    def forward(self, x, e):
        x = torch.cat([x, e], 1)
        x = F.relu(self.conv1_bn(self.conv1(x)), inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv2_bn(self.conv2(x)), inplace=True)

        g1 = self.channel_gate(x)
        g2 = self.spatial_gate(x)
        x = g1 + g2

        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x), inplace=True)
        dilate2_out = F.relu(self.dilate2(dilate1_out), inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out), inplace=True)

        out = x + dilate1_out + dilate2_out + dilate3_out
        return out


class UNetResNet34_128(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def load_pretrain(self):
        self.resnet.load_state_dict(torch.load('./models/pretrained_weights/resnet34-333f7ec4.pth'))

    def __init__(self):
        super(UNetResNet34_128, self).__init__()
        self.resnet = resnet34()

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = Dblock(512)

        self.decoder5 = DecoderV2(512, 256)
        self.decoder4 = DecoderV2(256 + 256, 256)
        self.decoder3 = DecoderV2(128 + 256, 128)
        self.decoder2 = DecoderV2(64 + 128, 128)

        self.logit = nn.Sequential(
            nn.Conv2d(64 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.stack([
            (x - mean[2]) / std[2],
            (x - mean[1]) / std[1],
            (x - mean[0]) / std[0],
        ], 1)  # （？，3, 128, 128)

        batch_size, C, H, W = x.shape

        e1 = self.conv1(x)  # (?,64,128,128)
        x = F.max_pool2d(e1, kernel_size=3, stride=2, padding=1)  # (?,64, 64, 64)

        e2 = self.encoder2(x)  # (?, 64,64,64)
        e3 = self.encoder3(e2)  # (?, 128,32,32)
        e4 = self.encoder4(e3)  # (?,256,16,16)
        e5 = self.encoder5(e4)  # (?, 512, 16,16)
        f = self.center(e5)  # (?, 512, 8, 8)

        d5 = self.decoder5(f)  # (?, 256, 16 ,16)
        d4 = self.decoder4(torch.cat([d5, e4], 1))  # (?, 256, 32, 32)
        d3 = self.decoder3(torch.cat([d4, e3], 1))  # (?, 128, 64, 64)
        d2 = self.decoder2(torch.cat([d3, e2], 1))  # (?, 128, 128, 128)
        d1 = torch.cat([d2, e1], 1)  # (?, 192, 128, 128)

        logit_pixel = self.logit(d1)  # (?, 1,128,128)

        return logit_pixel

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


class Deep_supervise_v2(nn.Module):
    def load_pretrain(self):
        self.resnet.load_state_dict(torch.load('./models/pretrained_weights/resnet34-333f7ec4.pth'))

    def __init__(self):
        super(Deep_supervise_v2, self).__init__()
        self.resnet = resnet34()

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = Dblock(512)

        self.decoder5 = DecoderV2(512, 256)
        self.decoder4 = DecoderV2(256 + 256, 256)
        self.decoder3 = DecoderV2(128 + 256, 128)
        self.decoder2 = DecoderV2(64 + 128, 128)

        self.fuse_pixel = nn.Sequential(
            ConvBn2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(128, 64, kernel_size=3, padding=1),
        )

        self.logit_pixel = nn.Sequential(
            ConvBn2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

        self.fuse_image = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
        )

        self.logit_image = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            ConvBn2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(64, 32, kernel_size=3, padding=1),
        )

        self.logit = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.stack([
            (x - mean[2]) / std[2],
            (x - mean[1]) / std[1],
            (x - mean[0]) / std[0],
        ], 1)  # （？，3, 128, 128)

        batch_size, C, H, W = x.shape

        e1 = self.conv1(x)  # (?,64,128,128)
        x = F.max_pool2d(e1, kernel_size=3, stride=2, padding=1)  # (?,64, 64, 64)

        e2 = self.encoder2(x)  # (?, 64,64,64)
        e3 = self.encoder3(e2)  # (?, 128,32,32)
        e4 = self.encoder4(e3)  # (?,256,16,16)
        e5 = self.encoder5(e4)  # (?, 512, 16,16)
        f = self.center(e5)  # (?, 512, 8, 8)

        d5 = self.decoder5(f)  # (?, 256, 16 ,16)
        d4 = self.decoder4(torch.cat([d5, e4], 1))  # (?, 256, 32, 32)
        d3 = self.decoder3(torch.cat([d4, e3], 1))  # (?, 128, 64, 64)
        d2 = self.decoder2(torch.cat([d3, e2], 1))  # (?, 128, 128, 128)
        d1 = torch.cat([d2, e1], 1)  # (?, 192, 128, 128)
        # loss 1
        fuse_pixel = self.fuse_pixel(d1)
        logit_pixel = self.logit_pixel(fuse_pixel)

        # # loss 2
        e = F.adaptive_avg_pool2d(e5, output_size=1)  # image pool
        e = F.dropout(e, p=0.50, training=self.training).view(batch_size, -1)
        fuse_image = self.fuse_image(e)

        logit_image = self.logit_image(fuse_image).view(-1)

        fuse = self.fuse(torch.cat([  # fuse
            fuse_pixel,  # 64 * 128 * 128
            F.upsample(fuse_image.view(batch_size, -1, 1, 1), scale_factor=128, mode='nearest')  # 64 * 128 * 128
        ], 1))  # 128**3

        logit = self.logit(fuse)

        return logit, logit_pixel, logit_image

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


if __name__ == '__main__':
    # net = UNetResNet34_128()
    net = Deep_supervise_v2()
    img = torch.rand((10, 128, 128))
    out = net(img)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
