import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

# Modified Cbam
# class Conv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                      padding, dilation, groups, bias)

#         self.count = 0
#         planes = in_channels
#         self.eps = 1e-5
#         ratio = 16

#         self.mlp = nn.Sequential(
#             nn.Conv2d(planes//groups, max(planes//16, 1), kernel_size=1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(max(planes//16, 1), planes//groups, kernel_size=1)
#         )

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.conv = nn.Conv2d(
#             2, 1, kernel_size=kernel_size, padding=kernel_size//2)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):

#         weight = self.weight

#         weight_avg = weight.mean(dim=1, keepdim=True).mean(
#             dim=2, keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_avg
#         std = weight.view(weight.size(
#             0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
#         weight = weight / std.expand_as(weight)

#         wght1 = self.avg_pool(weight)
#         wght1 = self.mlp(wght1)
#         wght2 = self.max_pool(weight)
#         wght2 = self.mlp(wght2)
#         wght = torch.sigmoid(wght1+wght2)
#         skip_1 = weight * wght
#         weight = weight + skip_1

#         # spatial attention
#         max_result, _ = torch.max(weight, dim=1, keepdim=True)
#         avg_result = torch.mean(weight, dim=1, keepdim=True)
#         result = torch.cat([max_result, avg_result], 1)
#         wght_2 = self.conv(result)
#         wght_2 = self.sigmoid(wght_2)
#         skip_2 = weight*wght_2

#         weight = weight + skip_1 + skip_2

#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)

# LWE


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

        self.count = 0
        planes = in_channels
        self.fc1 = nn.Conv2d(planes//groups, max(planes//16, 1), kernel_size=1)
        self.fc2 = nn.Conv2d(max(planes//16, 1), planes//groups, kernel_size=1)
        self.eps = 1e-5

    def forward(self, x):
        # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
        weight = self.weight

        # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
        weight_avg = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                           keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_avg
        std = weight.view(weight.size(
            0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)

        # LWE
        wght = F.avg_pool2d(weight, weight.size(2))
        wght = F.relu(self.fc1(wght))
        wght = F.sigmoid(self.fc2(wght))
        weight = weight * wght

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
