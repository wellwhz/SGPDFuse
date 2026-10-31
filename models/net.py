import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.color import luma, merge_color
from .dino import DINO, add_lora, lora_params
from .encoder import Encoder
from .decoder import Decoder
from .fusion import BaseFuse, DetailFuse, Emissivity

TASKS = {'ivif': (1, 3, 3), 'mff': (3, 3, 3), 'mef': (3, 3, 3), 'mif': (1, 1, 1)}


class SGPDFuse(nn.Module):
    def __init__(self, task='ivif', dim=64, blocks=4, dino='vitl16', dino_ckpt=None, lora=True, r=32, alpha=64, eps=None):
        super().__init__()
        self.task = task
        self.ca, self.cb, self.co = TASKS[task]
        self.dino = DINO(dino, dino_ckpt)
        if lora:
            add_lora(self.dino.vit, r, alpha)
        self.pa = nn.Conv2d(self.ca, 3, 1) if self.ca != 3 else nn.Identity()
        self.pb = nn.Conv2d(self.cb, 3, 1) if self.cb != 3 else nn.Identity()
        self.enc = Encoder(dim, blocks, self.dino.dim)
        self.dec1 = Decoder(dim * 2, 1)
        self.dec3 = Decoder(dim * 2, 3)
        self.fdec = Decoder(dim * 2, 1)
        self.eps = Emissivity(dim, 32, eps)
        self.bf = BaseFuse(dim)
        self.df = DetailFuse(dim)

    def encode(self, a, b):
        ba, da = self.enc(a, self.dino(self.pa(a)))
        bb, db = self.enc(b, self.dino(self.pb(b)))
        return ba, da, bb, db

    def recon(self, a, b):
        ba, da, bb, db = self.encode(a, b)
        ra = (self.dec1 if self.ca == 1 else self.dec3)(ba, da, a)
        rb = (self.dec1 if self.cb == 1 else self.dec3)(bb, db, b)
        return ra, rb

    def fuse(self, a, b):
        ba, da, bb, db = self.encode(a, b)
        eps = self.eps(ba, bb, da, db)
        eps = F.interpolate(eps, size=a.shape[-2:], mode='bilinear', align_corners=False)
        fb = self.bf(ba, bb, eps)
        fd = self.df(da, db, eps)
        if self.co == 1:
            return self.fdec(fb, fd), eps
        if self.task == 'ivif':
            ref = b
            y = self.fdec(fb, fd, luma(b))
        else:
            ref = eps * a + (1 - eps) * b
            y = self.fdec(fb, fd, eps * luma(a) + (1 - eps) * luma(b))
        return merge_color(y.clamp(0, 1), ref.clamp(0, 1)), eps

    def forward(self, a, b, stage=2):
        return self.recon(a, b) if stage == 1 else self.fuse(a, b)

    def train_params(self, stage):
        mods = [self.enc, self.dec1, self.dec3, self.fdec]
        if stage == 2:
            mods += [self.bf, self.df, self.eps]
        ps = lora_params(self.dino) + [p for m in mods for p in m.parameters()]
        for p in self.parameters():
            p.requires_grad = False
        for p in ps:
            p.requires_grad = True
        return ps
