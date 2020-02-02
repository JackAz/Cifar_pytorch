import torch.nn as nn
import torch.nn.functional as F

# Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

# out_size = (in_size+2*pad-filter)/strides+1
# cifar 32*32 , 50000 training images and 10000 testing images

class original(nn.Module):
    def __init__(self):
        super(original, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)

# class MobleNetV1(nn.Module):
#     def __init__(self, num_classes, grayscale=False):
#         dim = None
#         if grayscale == True:
#             dim = 1
#         else:
#             dim = 3
#         super(MobleNetV1, self).__init__()
#         self.mobilebone = nn.Sequential(
#             self._conv_bn(dim, 32, 2),
#             self._conv_dw(32, 64, 1),
#             self._conv_dw(64, 128, 2),
#             self._conv_dw(128, 128, 1),
#             self._conv_dw(128, 256, 2),
#             self._conv_dw(256, 256, 1),
#             self._conv_dw(256, 512, 2),
#             self._top_conv(512, 512, 5),
#             self._conv_dw(512, 1024, 2),
#             self._conv_dw(1024, 1024, 1),
#         )
#         self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
#         self.fc = nn.Linear(1024, num_classes)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, (2. / n) ** .5)
#             if isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#     def forward(self, x):
#         x = self.mobilebone(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         prob = F.softmax(x)
#         return x, prob
#     def _top_conv(self, in_channel, out_channel, blocks):
#         layers = []
#         for i in range(blocks):
#             layers.append(self._conv_dw(in_channel, out_channel, 1))
#         return nn.Sequential(*layers)
#     def _conv_bn(self, in_channel, out_channel, stride):
#         return nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3, stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#         )
#     def _conv_dw(self, in_channel, out_channel, stride):
#         return nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
#             nn.BatchNorm2d(in_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=False),
#         )