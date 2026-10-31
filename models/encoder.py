import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def gate(x):
    a, b = x.chunk(2, dim=1)
    return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c * 2
        self.conv1 = nn.Conv2d(c, d, 1)
        self.conv2 = nn.Conv2d(d, d, 3, 1, 1, groups=d)
        self.conv3 = nn.Conv2d(d // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
        self.conv4 = nn.Conv2d(c, d, 1)
        self.conv5 = nn.Conv2d(d // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = gate(self.conv2(self.conv1(self.norm1(x))))
        y = x + self.conv3(y * self.sca(y)) * self.beta
        return y + self.conv5(gate(self.conv4(self.norm2(y)))) * self.gamma


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))) + x)


def stem(cin, c):
    return nn.Sequential(nn.Conv2d(cin, c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))


class BaseEnc(nn.Module):
    def __init__(self, c=64, n=4):
        super().__init__()
        self.down1 = nn.Conv2d(c, c, 2, 2)
        self.blk1 = nn.Sequential(*[NAFBlock(c) for _ in range(n)])
        self.down2 = nn.Conv2d(c, c, 2, 2)
        self.blk2 = nn.Sequential(*[NAFBlock(c) for _ in range(n)])

    def forward(self, x):
        return self.blk2(self.down2(self.blk1(self.down1(x))))


class DetailEnc(nn.Module):
    def __init__(self, e=1024, c=64):
        super().__init__()
        act = nn.LeakyReLU(0.2, inplace=True)
        self.proj = nn.Sequential(nn.Conv2d(e, c * 16, 1), act, nn.Conv2d(c * 16, c * 16, 3, 1, 1), act)
        self.up1 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(c * 4, c * 4, 3, 1, 1), act)
        self.up2 = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(c, c, 3, 1, 1), act)
        self.refine = nn.Sequential(ResBlock(c), ResBlock(c))

    def forward(self, f):
        return self.refine(self.up2(self.up1(self.proj(f))))


class Encoder(nn.Module):
    def __init__(self, c=64, n=4, e=1024):
        super().__init__()
        self.p1 = stem(1, c)
        self.p3 = stem(3, c)
        self.base = BaseEnc(c, n)
        self.detail = DetailEnc(e, c)

    def forward(self, x, f):
        s = self.p1(x) if x.shape[1] == 1 else self.p3(x)
        return self.base(s), self.detail(f)
