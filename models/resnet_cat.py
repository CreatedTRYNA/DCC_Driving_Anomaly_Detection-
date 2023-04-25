import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from functools import partial
from ssmctb.torch_ssmctb_3d import SSMCTB
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


# ------------------3D CBAM --------------------------------
class Channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Channel_attention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.MLP = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        max_pool = self.max_pool(x).view([b, c])
        avg_pool = self.avg_pool(x).view([b, c])

        max_pool = self.MLP(max_pool)
        avg_pool = self.MLP(avg_pool)

        out = max_pool + avg_pool
        out = self.sigmoid(out).view([b, c, 1, 1, 1])
        return out * x


class Spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spacial_attention, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv3d(in_channels=2,
                              out_channels=1,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              bias=False
                              )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        out = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out * x


class CBAM_Block(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM_Block, self).__init__()
        self.channel_attention = Channel_attention(channel, ratio)
        self.spacial_attention = Spacial_attention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spacial_attention(x)
        return x


# ------------------3D CBAM --------------------------------

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
    if out.is_cuda:
        zero_pads = zero_pads.cuda()
    out = torch.cat([out, zero_pads], dim=1)

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, tracking=True, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.tracking = tracking
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=self.tracking)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=self.tracking)
        self.use_cbam = use_cbam
        self.cbam = CBAM_Block(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, tracking=True, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.tracking = tracking
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=self.tracking)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=self.tracking)
        self.use_cbam = use_cbam
        self.cbam = CBAM_Block(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 output_dim,
                 shortcut_type='B',
                 tracking=True, pre_train=False, use_cbam=False,use_ssmctb=True):
        self.inplanes = 64
        self.use_cbam = use_cbam
        self.use_ssmctb = use_ssmctb
        super(ResNet, self).__init__()
        if pre_train:
            self.conv1 = nn.Conv3d(
                3,
                64,
                kernel_size=7,
                stride=(1, 2, 2),
                padding=(3, 3, 3),
                bias=False)
        else:
            self.conv1 = nn.Conv3d(
                1,
                64,
                kernel_size=7,
                stride=(1, 2, 2),
                padding=(3, 3, 3),
                bias=False)
        self.tracking = tracking
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=self.tracking)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        # self.maxpool1 = nn.MaxPool3d(kernel_size=(last_duration, last_size, last_size), stride=1)
        self.ssmctb3d = SSMCTB(channels=128)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion, track_running_stats=self.tracking))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, tracking=self.tracking))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, tracking=self.tracking))

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

        # sig_x = torch.sigmoid(x) # sigmoid

        normed_x = F.normalize(x, p=2,
                               dim=1)  # normalized value, in order to eliminate influences of length during contrastive learning

        return x, normed_x


class ProjectionHead(nn.Module):
    def __init__(self, output_dim, model_depth):
        super(ProjectionHead, self).__init__()
        if model_depth == 18:
            # self.hidden = nn.Linear(150, 256)
            # self.hidden = nn.Linear(512, 256)
            self.hidden = nn.Linear(1024, 512)  # 给cat之后的特征使用
        elif model_depth == 50:
            self.hidden = nn.Linear(2048, 256)
        elif model_depth == 101:
            self.hidden = nn.Linear(2048, 256)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(512, output_dim)
        # self.out = nn.Linear(256, output_dim)
        self.fc = nn.Linear(output_dim,1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class ProjectionHeadwithBCE(nn.Module):
    def __init__(self, output_dim, model_depth):
        super(ProjectionHeadwithBCE, self).__init__()
        if model_depth == 18:
            # self.hidden = nn.Linear(150, 256)
            self.hidden = nn.Linear(512, 256)
        elif model_depth == 50:
            self.hidden = nn.Linear(2048, 256)
        elif model_depth == 101:
            self.hidden = nn.Linear(2048, 256)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(256, output_dim)
        self.fc = nn.Linear(output_dim,1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)
        x = F.normalize(x, p=2, dim=1)

        x_out = self.fc(x)
        score = self.sigmoid(x_out)
        return x,score.squeeze()


class LinearHead(nn.Module):
    def __init__(self, MLP_dim):
        super(LinearHead, self).__init__()
        self.hidden = nn.Linear(1024, MLP_dim)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(MLP_dim, 1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)

        return x


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet2D_18(**kwargs):

    torchvision.models.resnet18()
    pass
