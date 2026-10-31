import torch


def luma(x):
    if x.shape[1] == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def merge_color(y, ref):
    r, g, b = ref[:, 0:1], ref[:, 1:2], ref[:, 2:3]
    cb = -0.169 * r - 0.331 * g + 0.500 * b
    cr = 0.500 * r - 0.419 * g - 0.081 * b
    rgb = torch.cat([y + 1.403 * cr, y - 0.344 * cb - 0.714 * cr, y + 1.773 * cb], dim=1)
    return rgb.clamp(0, 1)
