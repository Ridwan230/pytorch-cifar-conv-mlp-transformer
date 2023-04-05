import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F


# MWE IMPLEMENTATION


# class Conv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                      padding, dilation, groups, bias)
#         self.count = 0
#         self.eps = 1e-5

#     def forward(self, x):
#         weight = self.weight

#         weight_avg = weight.mean(dim=1, keepdim=True).mean(dim=2,
#                                                            keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_avg
#         std = weight.view(weight.size(
#             0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
#         weight = weight / std.expand_as(weight)

#         # MWE
#         max_val = (1.0 + 0.1) * \
#             torch.max(torch.max(weight), -1 * torch.min(weight))
#         if self.count:
#             weight = (0.5*weight/max_val) / \
#                 torch.log10(1+torch.exp(-0.3*weight/max_val))
#         else:
#             self.count = 1
#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


# Modified Cbam


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

        self.count = 0
        planes = in_channels
        self.eps = 1e-5
        ratio = 16

        self.mlp = nn.Sequential(
            nn.Conv2d(planes//groups, max(planes//16, 1), kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(max(planes//16, 1), planes//groups, kernel_size=1)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        weight = self.weight
        weight_avg = weight.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_avg
        std = weight.view(weight.size(
            0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        weight = weight / std.expand_as(weight)

        wght1 = self.avg_pool(weight)
        wght1 = self.mlp(wght1)
        wght2 = self.max_pool(weight)
        wght2 = self.mlp(wght2)
        wght = torch.sigmoid(wght1+wght2)
        skip_1 = weight * wght
        weight = weight + skip_1

        # spatial attention
        max_result, _ = torch.max(weight, dim=1, keepdim=True)
        avg_result = torch.mean(weight, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        wght_2 = self.conv(result)
        wght_2 = self.sigmoid(wght_2)
        skip_2 = weight*wght_2

        weight = weight + skip_1 + skip_2

        # Triple Attn

        # weight2 = weight
        # x = torch.cat((torch.max(weight2, 1)[0].unsqueeze(
        #     1), torch.mean(weight2, 1).unsqueeze(1)), dim=1)
        # x = self.conv(x)
        # x = torch.sigmoid_(x)

        # weight3 = weight
        # x_perm1 = weight3.permute(0, 2, 1, 3).contiguous()
        # x_perm1 = torch.cat((torch.max(x_perm1, 1)[0].unsqueeze(
        #     1), torch.mean(x_perm1, 1).unsqueeze(1)), dim=1)
        # x_perm1 = self.conv(x_perm1)
        # x_perm1 = torch.sigmoid_(x_perm1)
        # x_perm1 = x_perm1.permute(0, 2, 1, 3).contiguous()

        # weight4 = weight
        # x_perm2 = weight4.permute(0, 3, 2, 1).contiguous()
        # x_perm2 = torch.cat((torch.max(x_perm2, 1)[0].unsqueeze(
        #     1), torch.mean(x_perm2, 1).unsqueeze(1)), dim=1)
        # x_perm2 = self.conv(x_perm2)
        # x_perm2 = torch.sigmoid_(x_perm2)
        # x_perm2 = x_perm2.permute(0, 3, 2, 1).contiguous()

        # weight = weight * ((1/3)*(x+x_perm1+x_perm2))

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# LWE


# class Conv2d(nn.Conv2d):

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                      padding, dilation, groups, bias)

#         self.count = 0
#         planes = in_channels
#         self.fc1 = nn.Conv2d(planes//groups, max(planes//16, 1), kernel_size=1)
#         self.fc2 = nn.Conv2d(max(planes//16, 1), planes//groups, kernel_size=1)
#         self.eps = 1e-5

#     def forward(self, x):
#         # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
#         weight = self.weight

#         # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
#         weight_avg = weight.mean(dim=1, keepdim=True).mean(dim=2,
#                                                            keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_avg
#         std = weight.view(weight.size(
#             0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
#         weight = weight / std.expand_as(weight)

#         # LWE
#         wght = F.avg_pool2d(weight, weight.size(2))
#         wght = F.relu(self.fc1(wght))
#         wght = F.sigmoid(self.fc2(wght))
#         weight = weight * wght

#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


# class ZPool(nn.Module):
#     def forward(self, x):
#         return torch.cat(
#             (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
#         )


# class BasicConv(nn.Module):
#     def __init__(
#         self,
#         in_planes,
#         out_planes,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         relu=True,
#         bn=True,
#         bias=False,
#     ):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(
#             in_planes,
#             out_planes,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#         )
#         self.bn = (
#             nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
#             if bn
#             else None
#         )
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x


# class AttentionGate(nn.Module):
#     def __init__(self):
#         super(AttentionGate, self).__init__()
#         kernel_size = 3
#         self.compress = ZPool()
#         self.conv = BasicConv(
#             2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
#         )

#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.conv(x_compress)
#         scale = torch.sigmoid_(x_out)
#         return x * scale


# class TripletAttention(nn.Module):
#     def __init__(self, no_spatial=False):
#         super(TripletAttention, self).__init__()
#         self.cw = AttentionGate()
#         self.hc = AttentionGate()

#     def forward(self, x):
#         x_perm1 = x.permute(0, 2, 1, 3).contiguous()
#         x_out1 = self.cw(x_perm1)
#         x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
#         x_perm2 = x.permute(0, 3, 2, 1).contiguous()
#         x_out2 = self.hc(x_perm2)
#         x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
#         return (1 / 2 * (x_out11 + x_out21))


# class Conv2d(nn.Conv2d):

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                      padding, dilation, groups, bias)

#         self.count = 0
#         planes = in_channels
#         self.fc1 = nn.Conv2d(planes//groups, max(planes//16, 1), kernel_size=1)
#         self.fc2 = nn.Conv2d(max(planes//16, 1), planes//groups, kernel_size=1)
#         self.eps = 1e-5
#         self.conv = nn.Conv2d(2, 1, kernel_size=1)

#     def forward(self, x):
#         # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
#         weight = self.weight

#         # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
#         weight_avg = weight.mean(dim=1, keepdim=True).mean(dim=2,
#                                                            keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_avg
#         std = weight.view(weight.size(
#             0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
#         weight = weight / std.expand_as(weight)

#         # Triple Attn
#         wght=weight
#         # print("1")
#         # print(wght.shape)
#         x_perm1 = wght.permute(0, 2, 1, 3).contiguous()
#         # print("2")
#         # print(x_perm1.shape)
#         x_perm1 = torch.cat((torch.max(x_perm1, 1)[0].unsqueeze(1), torch.mean(x_perm1, 1).unsqueeze(1)), dim=1)
#         # print("3")
#         # print(x_perm1.shape)
#         x_perm1=self.conv(x_perm1)
#         # print("4")
#         # print(x_perm1.shape)
#         x_perm1=torch.sigmoid_(x_perm1)
#         # print("5")
#         # print(x_perm1.shape)
#         x_perm1=x_perm1.permute(0, 2, 1, 3).contiguous()
#         # print("6")
#         # print(x_perm1.shape)
#         # temp = wght.permute(0, 2, 1, 3).contiguous()
#         # print("7")
#         # print(temp.shape)
#         # x_perm1 = temp*x_perm1

#         weight_avg=weight
#         # print("1")
#         # print(wght.shape)
#         x_perm2 = wght.permute(0, 2, 1, 3).contiguous()
#         # print("2")
#         # print(x_perm2.shape)
#         x_perm2 = torch.cat((torch.max(x_perm2, 1)[0].unsqueeze(1), torch.mean(x_perm2, 1).unsqueeze(1)), dim=1)
#         # print("3")
#         # print(x_perm2.shape)
#         x_perm2=self.conv(x_perm2)
#         # print("4")
#         # print(x_perm2.shape)
#         x_perm2=torch.sigmoid_(x_perm2)
#         # print("5")
#         # print(x_perm2.shape)
#         x_perm2=x_perm2.permute(0, 2, 1, 3).contiguous()
#         # print("6")
#         # print(x_perm2.shape)


#         weight = weight * (.5*(x_perm1+x_perm2))


#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


# class Conv2d(nn.Conv2d):

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                      padding, dilation, groups, bias)

#         self.count = 0
#         planes = in_channels
#         self.fc1 = nn.Conv2d(planes//groups, max(planes//16, 1), kernel_size=1)
#         self.fc2 = nn.Conv2d(max(planes//16, 1), planes//groups, kernel_size=1)
#         self.eps = 1e-5
#         self.conv = nn.Conv2d(
#             planes//groups,
#             max(planes//16, 1),
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#         )

#     def forward(self, x):
#         # return super(Conv2d, self).forward(x) #use this for normal convolution without any WE
#         weight = self.weight

#         # Following 4 lines are implementation of weight standardization. Optional. Seems to work well, except for depthwise convolution. See paper for more details.
#         weight_avg = weight.mean(dim=1, keepdim=True).mean(dim=2,
#                                                            keepdim=True).mean(dim=3, keepdim=True)
#         weight = weight - weight_avg
#         std = weight.view(weight.size(
#             0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
#         weight = weight / std.expand_as(weight)

#         # Triple Attn
#         wght=weight
#         x_perm1 = wght.permute(0, 2, 1, 3).contiguous()
#         x_perm1 = torch.cat((torch.max(x_perm1, 1)[0].unsqueeze(1), torch.mean(x_perm1, 1).unsqueeze(1)), dim=1)
#         x_perm1=self.conv(x_perm1)
#         x_perm1=torch.sigmoid_(x_perm1)
#         x_perm1 = wght.permute(0, 2, 1, 3).contiguous()*x_perm1
#         x_perm1 = wght.permute(0, 2, 1, 3).contiguous()

#         weight = weight * x_perm1
#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
