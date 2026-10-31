import argparse
import os
from datetime import datetime

import torch
import torch.distributed as dist
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import SGPDFuse
from utils.dataset import FusionData
from utils.loss import FusionLoss, ReconLoss


def setup():
    if 'RANK' not in os.environ:
        return 0, 1, 0
    dist.init_process_group('nccl')
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    return rank, world, local


def mean(v, world):
    if world > 1:
        t = torch.tensor(v, device='cuda')
        dist.all_reduce(t)
        v = t.item() / world
    return v


def run_epoch(model, loader, crit, opt, scaler, stage, ep, writer, rank, world, log_freq):
    model.train()
    total, logs = 0.0, {}
    bar = tqdm(loader, desc=f'[S{stage}] {ep}', disable=rank != 0)
    for i, (a, b) in enumerate(bar):
        a, b = a.cuda(non_blocking=True), b.cuda(non_blocking=True)
        opt.zero_grad()
        with autocast('cuda', enabled=scaler.is_enabled()):
            if stage == 1:
                ra, rb = model(a, b, stage=1)
                loss, log = crit(ra, rb, a, b)
            else:
                f, eps = model(a, b)
                loss, log = crit(a, b, f)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        for k, v in log.items():
            logs[k] = logs.get(k, 0) + v
        if rank == 0:
            bar.set_postfix(loss=f'{loss.item():.4f}')
            if i % log_freq == 0:
                step = ep * len(loader) + i
                for k, v in log.items():
                    writer.add_scalar(f'S{stage}/{k}', v, step)
                writer.add_image(f'S{stage}/a', a[0].clamp(0, 1), step)
                writer.add_image(f'S{stage}/b', b[0].clamp(0, 1), step)
                if stage == 1:
                    writer.add_image('S1/ra', ra[0].detach().clamp(0, 1), step)
                    writer.add_image('S1/rb', rb[0].detach().clamp(0, 1), step)
                else:
                    writer.add_image('S2/fused', f[0].detach().clamp(0, 1), step)
                    writer.add_image('S2/eps', eps[0].detach().clamp(0, 1), step)
    n = len(loader)
    return mean(total / n, world), {k: v / n for k, v in logs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/IVIF.yml')
    with open(parser.parse_args().config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    tc = cfg['train']
    rank, world, local = setup()

    save_dir = os.path.join(tc['save_dir'], f"{cfg['task']}_{datetime.now():%Y%m%d_%H%M%S}")
    writer = None
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    net = SGPDFuse(cfg['task'], **cfg['model']).cuda()
    if tc['resume']:
        net.load_state_dict(torch.load(tc['resume'], map_location='cpu')['model'], strict=False)
    model = DDP(net, device_ids=[local], find_unused_parameters=True) if world > 1 else net

    data = FusionData(cfg['data']['root'], cfg['data']['a'], cfg['data']['b'], net.ca, net.cb)
    sampler = DistributedSampler(data) if world > 1 else None
    loader = DataLoader(data, tc['batch_size'], shuffle=sampler is None, sampler=sampler, num_workers=tc['workers'], pin_memory=True, drop_last=True)
    scaler = GradScaler('cuda', enabled=tc['amp'])

    for stage in (1, 2):
        if stage < (tc['resume_stage'] or 1):
            continue
        epochs = tc['epochs'][stage - 1]
        if stage == 1:
            crit = ReconLoss().cuda()
        else:
            net.fdec.load_state_dict(net.dec1.state_dict())
            crit = FusionLoss().cuda()
        lr = tc['lr'] if stage == 1 else tc['lr'] * 0.5
        opt = torch.optim.AdamW(net.train_params(stage), lr=lr, weight_decay=tc['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        for ep in range(epochs):
            if sampler:
                sampler.set_epoch(ep)
            loss, logs = run_epoch(model, loader, crit, opt, scaler, stage, ep, writer, rank, world, tc['log_freq'])
            sched.step()
            if rank == 0:
                print(f'[S{stage}] epoch {ep}: ' + ', '.join(f'{k}={v:.4f}' for k, v in logs.items()))
                writer.add_scalar(f'S{stage}/epoch_loss', loss, ep)
                writer.add_scalar(f'S{stage}/lr', sched.get_last_lr()[0], ep)
                if (ep + 1) % tc['save_freq'] == 0:
                    torch.save({'stage': stage, 'epoch': ep, 'model': net.state_dict()}, os.path.join(save_dir, 'checkpoints', f's{stage}_e{ep + 1}.pth'))
        if rank == 0:
            torch.save({'stage': stage, 'epoch': epochs - 1, 'model': net.state_dict()}, os.path.join(save_dir, f'stage{stage}.pth'))

    if rank == 0:
        torch.save(net.state_dict(), os.path.join(save_dir, 'final.pth'))
        writer.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
