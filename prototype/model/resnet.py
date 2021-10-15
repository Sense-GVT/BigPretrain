import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import spring.linklink as link
from spring.linklink.nn import SyncBatchNorm2d
from spring.linklink.nn import syncbnVarMode_t
from prototype.utils.dist import simple_group_split
from prototype.utils.misc import get_logger
BN = None

__all__ = ['resnet18', 'resnet26', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet_custom']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    """Redidual Networks class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_
    """
    def __init__(self,
                 block,
                 layers,
                 inplanes=64,
                 deep_stem=False,
                 avg_down=False,
                 bypass_last_bn=False,
                 nnie_type=False,
                 scale=1.0,
                 bn_group_size=1,
                 bn_var_mode=syncbnVarMode_t.L2,
                 bn_sync_stats=True,
                 use_sync_bn=True,
                 return_fpn=True,
                 **kwargs):
        r"""
        Arguments:

        - layers (:obj:`list` of 4 ints): how many layers in each stage
        - deep_stem (:obj:`bool`): whether to use deep_stem as the first conv
        - avg_down (:obj:`bool`): whether to use avg_down when spatial downsample
        - bypass_last_bn (:obj:`bool`): whether use bypass_last_bn
        - bn (:obj:`dict`): definition of batchnorm
        """

        super(ResNet, self).__init__()

        logger = get_logger(__name__)

        global BN
        rank = link.get_rank()
        world_size = link.get_world_size()
        bn_group = simple_group_split(world_size, rank, world_size // bn_group_size)

        def BNFunc(*args, **kwargs):
            return SyncBatchNorm2d(*args, **kwargs,
                                   group=bn_group,
                                   sync_stats=bn_sync_stats,
                                   var_mode=bn_var_mode)


        if use_sync_bn:
            BN = BNFunc
            print('[Rank {}] use SyncBatchNorm in bn_group {}'.format(rank, bn_group), flush=True)
        else:
            BN = nn.BatchNorm2d
        self.return_fpn = return_fpn
        self.inplanes = int(inplanes * scale)
        self.deep_stem = deep_stem
        self.avg_down = avg_down

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, self.inplanes // 2, kernel_size=3, stride=2, padding=1, bias=False),
                        BN(self.inplanes // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.inplanes // 2, self.inplanes // 2, kernel_size=3,
                                  stride=1, padding=1, bias=False),
                        BN(self.inplanes // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.inplanes // 2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if nnie_type:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 * scale), layers[0])
        self.layer2 = self._make_layer(block, int(128 * scale), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * scale), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * scale), layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.init_weights()


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (SyncBatchNorm2d,_BatchNorm, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, freeze_backbone=False, freeze_at=0):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1)]:
                x = self.relu(bn(conv(x)))
            x = self.maxpool(x)
            return x
        # set_trace()
        if freeze_backbone and freeze_at == 4:
            with torch.no_grad():
                x = x.type(self.conv1.weight.dtype)
                x = stem(x)
                x = self.layer1(x)
                if self.return_fpn: out1 = x
                x = self.layer2(x)
                if self.return_fpn: out2 = x
                x = self.layer3(x)
                if self.return_fpn: out3 = x
            x = self.layer4(x)
            if self.return_fpn: out4 = x
        elif freeze_backbone:
            with torch.no_grad():
                x = x.type(self.conv1.weight.dtype)
                x = stem(x)
                x = self.layer1(x)
                if self.return_fpn: out1 = x
                x = self.layer2(x)
                if self.return_fpn: out2 = x
                x = self.layer3(x)
                if self.return_fpn: out3 = x
                x = self.layer4(x)
                if self.return_fpn: out4 = x
        elif not freeze_backbone:
            x = x.type(self.conv1.weight.dtype)
            x = stem(x)
            x = self.layer1(x)
            if self.return_fpn: out1 = x
            x = self.layer2(x)
            if self.return_fpn: out2 = x
            x = self.layer3(x)
            if self.return_fpn: out3 = x
            x = self.layer4(x)
            if self.return_fpn: out4 = x
        else:
            assert False, print(freeze_backbone,freeze_at)

        if self.return_fpn:
            return out1, out2, out3, out4

        x = self.avgpool(x)
        return x


def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet26(**kwargs):
    """
    Constructs a ResNet-26 model.
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """
    Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """
    Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """
    Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet_custom(**kwargs):
    """
    Constructs a custom ResNet model with custom block and depth.
    """
    assert 'block' in kwargs and 'layers' in kwargs, 'Require block and layers'
    block = kwargs.pop('block')
    layers = kwargs.pop('layers')
    if block == 'basic':
        block = BasicBlock
    elif block == 'bottleneck':
        block = Bottleneck
    else:
        raise Exception('Unsupported block type.')
    model = ResNet(block, layers, **kwargs)
    return model
