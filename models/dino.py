import math

import torch
import torch.nn as nn
from dinov3.hub import backbones

DIMS = {'vits16': 384, 'vitb16': 768, 'vitl16': 1024, 'vitl16plus': 1024}


class LoRA(nn.Module):
    def __init__(self, cin, cout, r=8, alpha=16):
        super().__init__()
        self.s = alpha / r
        self.a = nn.Parameter(torch.zeros(r, cin))
        self.b = nn.Parameter(torch.zeros(cout, r))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return x @ self.a.T @ self.b.T * self.s


class QKV(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        c = base.out_features // 3
        self.q = LoRA(base.in_features, c, r, alpha)
        self.k = LoRA(base.in_features, c, r, alpha)
        self.v = LoRA(base.in_features, c, r, alpha)
        for p in base.parameters():
            p.requires_grad = False

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        q, k, v = self.base(x).chunk(3, dim=-1)
        return torch.cat([q + self.q(x), k + self.k(x), v + self.v(x)], dim=-1)


def add_lora(vit, r=32, alpha=64):
    for blk in vit.blocks:
        blk.attn.qkv = QKV(blk.attn.qkv, r, alpha)


def lora_params(m):
    return [p for x in m.modules() if isinstance(x, LoRA) for p in x.parameters()]


class DINO(nn.Module):
    def __init__(self, name='vitl16', ckpt=None):
        super().__init__()
        self.dim = DIMS[name]
        self.vit = getattr(backbones, f'dinov3_{name}')(pretrained=False)
        if ckpt:
            self.vit.load_state_dict(torch.load(ckpt, map_location='cpu'))

    def forward(self, x):
        b, _, h, w = x.shape
        t = self.vit.forward_features(x)['x_norm_patchtokens']
        return t.reshape(b, h // 16, w // 16, -1).permute(0, 3, 1, 2)
