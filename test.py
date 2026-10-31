import argparse
import os

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models import SGPDFuse
from utils.img_read_save import img_read, img_save

EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def pad16(x):
    h, w = x.shape[-2:]
    return F.pad(x, (0, -w % 16, 0, -h % 16), mode='reflect')


def test(cfg, ckpt, data, out, save_eps=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mc = dict(cfg['model'])
    if mc.get('dino_ckpt') and not os.path.isfile(mc['dino_ckpt']):
        mc['dino_ckpt'] = None
    model = SGPDFuse(cfg['task'], **mc)
    sd = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(sd.get('model', sd), strict=False)
    model = model.to(device).eval()

    da, db = os.path.join(data, cfg['data']['a']), os.path.join(data, cfg['data']['b'])
    stems = {os.path.splitext(n)[0]: n for n in os.listdir(db) if n.lower().endswith(EXTS)}
    names = sorted(n for n in os.listdir(da) if n.lower().endswith(EXTS) and os.path.splitext(n)[0] in stems)
    os.makedirs(out, exist_ok=True)

    with torch.no_grad():
        for n in tqdm(names):
            stem = os.path.splitext(n)[0]
            a = img_read(os.path.join(da, n), gray=model.ca == 1).to(device)
            b = img_read(os.path.join(db, stems[stem]), gray=model.cb == 1).to(device)
            h, w = a.shape[-2:]
            f, eps = model(pad16(a), pad16(b))
            img_save(f[..., :h, :w], os.path.join(out, stem + '.png'))
            if save_eps:
                e = eps[..., :h, :w]
                img_save((e - e.min()) / (e.max() - e.min() + 1e-8), os.path.join(out, stem + '_eps.png'))
    print(f'{len(names)} results saved in {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/IVIF.yml')
    parser.add_argument('--ckpt', type=str, default='ckpt/IVIF.pth')
    parser.add_argument('--data', type=str, default='datasets/M3FD/test')
    parser.add_argument('--out', type=str, default='results/IVIF')
    parser.add_argument('--eps', action='store_true')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    test(cfg, args.ckpt, args.data, args.out, args.eps)
