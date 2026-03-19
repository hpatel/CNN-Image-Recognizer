import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride,
            1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64,2,stride=1)
        self.layer2 = self._make_layer(128,2,stride=2)
        self.layer3 = self._make_layer(256,2,stride=2)
        self.layer4 = self._make_layer(512,2,stride=2)

        self.linear = nn.Linear(512,num_classes)

    def _make_layer(self,out_channels,blocks,stride):

        strides = [stride] + [1]*(blocks-1)
        layers = []

        for s in strides:
            layers.append(
                BasicBlock(self.in_channels,out_channels,s)
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self,x):

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,4)

        x = x.view(x.size(0),-1)

        x = self.linear(x)

        return x