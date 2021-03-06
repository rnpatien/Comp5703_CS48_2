'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from models.gan.base import BaseDiscriminator

class prune_conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(prune_conv2d, self).__init__(*args, **kwargs)
        self.prune_mask = torch.ones(list(self.weight.shape)) #rp??
        self.prune_flag = False
        return

    def forward(self, input):
        if not self.prune_flag:
            weight = self.weight
        else:
            weight = self.weight * self.prune_mask.to('cuda')
        return self._conv_forward(input, weight, None)

    def set_prune_flag(self, flag):
        self.prune_flag = flag


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return prune_conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
            )
        self.prune_flag = False

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.leaky_relu(out, 0.1, inplace=True)
        return out

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in [self.conv1, self.conv2]:
            module.set_prune_flag(flag)


class SNResNet(BaseDiscriminator):
    def __init__(self, block, num_blocks, n_classes=1, disable_sn=False, **kwargs):
        self.in_planes = 64
        self.n_features = 512 * block.expansion
        super(SNResNet, self).__init__(self.n_features, n_classes=n_classes, **kwargs)
        self.disable_sn = disable_sn

        self.conv1 = conv3x3(3, 64)
        # self.conv1 = prune_conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if not disable_sn:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    spectral_norm(m)
                elif isinstance(m, nn.Linear):
                    spectral_norm(m)
                elif isinstance(m, nn.Embedding):
                    spectral_norm(m)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def penultimate(self, x):
        # print("penult")
        out = x * 2. - 1.
        out = self.conv1(out)
        out = F.leaky_relu(out, 0.1, inplace=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return out

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in [self.conv1,]:
            module.set_prune_flag(flag)
        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in stage:
                if isinstance(layer, BasicBlock)  or isinstance(layer, prune_conv2d):
                    layer.set_prune_flag(flag)

def D_SNResNet18(**kwargs):
    return SNResNet(BasicBlock, [2,2,2,2], **kwargs)

def D_SNResNet34(**kwargs):
    return SNResNet(BasicBlock, [3,4,6,3], **kwargs)
