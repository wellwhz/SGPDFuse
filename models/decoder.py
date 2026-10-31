import torch
import torch.nn as nn

from .encoder import ResBlock


class Decoder(nn.Module):
    def __init__(self, cin=128, cout=3):
        super().__init__()
        c = cin // 4
        self.up1 = nn.Sequential(nn.Conv2d(cin, cin, 3, 1, 1), nn.PixelShuffle(2))
        self.blk1 = nn.Sequential(ResBlock(c), ResBlock(c))
        self.up2 = nn.Sequential(nn.Conv2d(c, c * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.blk2 = nn.Sequential(ResBlock(c), ResBlock(c))
        self.out = nn.Conv2d(c, cout, 3, 1, 1)
        self.scale = nn.Parameter(torch.tensor(0.5))
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        nn.init.zeros_(self.out.bias)

    def forward(self, base, detail, skip=None):
        x = self.blk2(self.up2(self.blk1(self.up1(torch.cat([base, detail], dim=1)))))
        x = self.out(x)
        if skip is None:
            return torch.sigmoid(x)
        return torch.clamp(skip + self.scale.clamp(0.1, 1.0) * torch.tanh(x), 0, 1)
