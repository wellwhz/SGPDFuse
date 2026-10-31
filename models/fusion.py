import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import NAFBlock


class GNBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        return self.act(self.norm2(self.conv2(y)) + x)


class BaseFuse(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c * 2, c, 1), nn.LeakyReLU(0.2, inplace=True), NAFBlock(c), NAFBlock(c))
        self.attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c * 2, c, 1), nn.ReLU(inplace=True), nn.Conv2d(c, 2, 1), nn.Softmax(dim=1))
        self.k = nn.Parameter(torch.tensor(0.3))

    def forward(self, a, b, eps):
        w = self.attn(torch.cat([a, b], dim=1))
        d = self.k * (F.adaptive_avg_pool2d(eps, 1) - 0.5)
        f = torch.clamp(w[:, :1] + d, 0.1, 0.9) * a + torch.clamp(w[:, 1:] - d, 0.1, 0.9) * b
        return self.conv(torch.cat([f, f], dim=1))


class DetailFuse(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c * 2, c, 1), nn.LeakyReLU(0.2, inplace=True), GNBlock(c), GNBlock(c))
        self.refine = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.k = nn.Parameter(torch.tensor(0.3))

    def forward(self, a, b, eps):
        la = a.abs().sum(1, keepdim=True)
        lb = b.abs().sum(1, keepdim=True)
        s = la + lb + 1e-8
        eps = F.interpolate(eps, size=a.shape[-2:], mode='bilinear', align_corners=False)
        d = self.k * (eps - 0.5)
        f = torch.clamp(la / s + d, 0, 1) * a + torch.clamp(lb / s - d, 0, 1) * b
        return self.refine(self.conv(torch.cat([f, f], dim=1)))


def eps_range(path):
    if path is None:
        return 0.1, 1.0
    with open(path, 'r', encoding='utf-8') as f:
        v = list(json.load(f).values())
    return min(v) * 0.9, min(max(v) * 1.1, 1.0)


class Emissivity(nn.Module):
    def __init__(self, c=64, h=32, path=None):
        super().__init__()
        lo, hi = eps_range(path)
        self.base = nn.Sequential(nn.Conv2d(c * 2, h, 3, 1, 1), nn.GroupNorm(8, h), nn.SiLU(inplace=True))
        self.detail = nn.Sequential(nn.Conv2d(c * 2, h, 3, 1, 1), nn.GroupNorm(8, h), nn.SiLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(h * 2, h, 3, 1, 1), nn.GroupNorm(8, h), nn.SiLU(inplace=True), nn.Conv2d(h, 1, 1), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(hi - lo))
        self.bias = nn.Parameter(torch.tensor(lo))

    def forward(self, ba, bb, da, db):
        size = ba.shape[-2:]
        da = F.interpolate(da, size=size, mode='bilinear', align_corners=False)
        db = F.interpolate(db, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([self.base(torch.cat([ba, bb], 1)), self.detail(torch.cat([da, db], 1))], 1)
        return self.head(x) * self.scale + self.bias
