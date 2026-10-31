import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import luma


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer('kx', k.view(1, 1, 3, 3))
        self.register_buffer('ky', k.t().contiguous().view(1, 1, 3, 3))

    def forward(self, x):
        x = luma(x)
        return F.conv2d(x, self.kx.to(x.dtype), padding=1).abs() + F.conv2d(x, self.ky.to(x.dtype), padding=1).abs()


class SSIM(nn.Module):
    def __init__(self, size=11):
        super().__init__()
        self.size = size
        g = torch.exp(-((torch.arange(size, dtype=torch.float32) - size // 2) ** 2) / 8)
        g = g / g.sum()
        self.register_buffer('win', g.outer(g).view(1, 1, size, size))

    def forward(self, x, y):
        c = x.shape[1]
        w = self.win.to(x.dtype).expand(c, -1, -1, -1)
        p = self.size // 2
        conv = lambda t: F.conv2d(t, w, padding=p, groups=c)
        mx, my = conv(x), conv(y)
        sx = conv(x * x) - mx ** 2
        sy = conv(y * y) - my ** 2
        sxy = conv(x * y) - mx * my
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx.clamp(min=0) + sy.clamp(min=0) + c2) + 1e-8)
        return 1 - s.mean()


class ReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def forward(self, ra, rb, a, b):
        ra, rb = ra.clamp(0, 1), rb.clamp(0, 1)
        la = F.l1_loss(ra, a) + 0.5 * self.ssim(ra, a)
        lb = F.l1_loss(rb, b) + 0.5 * self.ssim(rb, b)
        lc = sum(F.l1_loss(rb[:, i:i + 1], b[:, i:i + 1]) for i in range(3)) if rb.shape[1] == 3 else 0
        return la + lb + lc, {'a': la.item(), 'b': lb.item()}


class FusionLoss(nn.Module):
    def __init__(self, w_tv=5.0):
        super().__init__()
        self.w_tv = w_tv
        self.ssim = SSIM()
        self.sobel = Sobel()

    def forward(self, a, b, f):
        f = f.clamp(0, 1)
        ya, yb, yf = luma(a), luma(b), luma(f)
        l_int = F.l1_loss(yf, torch.max(ya, yb))
        l_grad = F.l1_loss(self.sobel(yf), torch.max(self.sobel(ya), self.sobel(yb)))
        l_b = 5 * self.ssim(yb, yf) + F.mse_loss(yb, yf)
        l_a = 5 * self.ssim(ya, yf) + F.mse_loss(ya, yf)
        l_tv = (f[:, :, 1:] - f[:, :, :-1]).abs().mean() + (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean()
        loss = l_int + 10 * l_grad + l_a + l_b + self.w_tv * l_tv
        return loss, {'int': l_int.item(), 'grad': l_grad.item(), 'ssim_a': l_a.item(), 'ssim_b': l_b.item(), 'tv': l_tv.item(), 'total': loss.item()}
