import numpy as np
import torch
from PIL import Image


def img_read(path, gray=False):
    x = np.asarray(Image.open(path).convert('L' if gray else 'RGB'), dtype=np.float32) / 255.0
    x = torch.from_numpy(x)
    return x[None, None] if gray else x.permute(2, 0, 1)[None]


def img_save(x, path):
    x = (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(x[..., 0] if x.shape[-1] == 1 else x).save(path)
