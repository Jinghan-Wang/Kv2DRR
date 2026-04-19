import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.unpaired_nii_dataset import UnpairedNiiDataset, ValNiiDataset
from models.networks import ResidualToneMapper, NLayerDiscriminator
from utils.image_ops import make_fused_pred_from_pred_input
from utils.losses import GANLoss, GradientLoss, TVLoss, SSIMLoss
from utils.misc import seed_everything, load_yaml, save_yaml, ensure_dir, copy_file, AverageMeter
from utils.nii_io import load_nii_image, save_array_as_nii



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()



def build_model(cfg, device):
    model = ResidualToneMapper(
        in_channels=cfg['input']['in_channels'],
        base_channels=cfg['model']['base_channels'],
        num_down=cfg['model']['num_down'],
        num_res_blocks=cfg['model']['num_res_blocks'],
        delta_scale=cfg['model']['delta_scale'],
        use_tanh=cfg['model']['use_tanh'],
    ).to(device)

    disc = NLayerDiscriminator(
        in_channels=cfg['input']['out_channels'],
        base_channels=cfg['discriminator']['base_channels'],
    ).to(device)

    return model, disc



def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6



def train_one_epoch(model, disc, loader, opt_g, opt_d, scaler, losses_obj, device, cfg, epoch):
    model.train()
    disc.train()

    meter_total = AverageMeter()
    meter_adv = AverageMeter()
    meter_id = AverageMeter()
    meter_grad = AverageMeter()
    meter_ssim = AverageMeter()
    meter_delta_l1 = AverageMeter()
    meter_delta_tv = AverageMeter()
    meter_d = AverageMeter()

    pbar = tqdm(loader, desc=f'Train {epoch:03d}')
    for it, batch in enumerate(pbar, start=1):
        kv = batch['kv'].to(device, non_blocking=True)
        drr = batch['drr'].to(device, non_blocking=True)

        # ----------------------
        # Train D
        # ----------------------
        opt_d.zero_grad(set_to_none=True)
        with autocast(enabled=cfg['train']['use_amp']):
            fake = model(kv)['pred'].detach()
            pred_real = disc(drr)
            pred_fake = disc(fake)
            loss_d_real = losses_obj['gan'](pred_real, True)
            loss_d_fake = losses_obj['gan'](pred_fake, False)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        scaler.scale(loss_d).backward()
        scaler.step(opt_d)

        # ----------------------
        # Train G
        # ----------------------
        opt_g.zero_grad(set_to_none=True)
        with autocast(enabled=cfg['train']['use_amp']):
            out_kv = model(kv)
            fake = out_kv['pred']
            delta = out_kv['delta']

            out_drr = model(drr)
            idt = out_drr['pred']

            pred_fake_for_g = disc(fake)
            loss_adv = losses_obj['gan'](pred_fake_for_g, True)
            loss_id = F.l1_loss(idt, drr)
            loss_grad = losses_obj['grad'](fake, kv)
            loss_ssim = losses_obj['ssim'](fake, kv)
            loss_delta_l1 = torch.mean(torch.abs(delta))
            loss_delta_tv = losses_obj['tv'](delta)

            total = (
                cfg['loss']['lambda_adv'] * loss_adv +
                cfg['loss']['lambda_id'] * loss_id +
                cfg['loss']['lambda_grad'] * loss_grad +
                cfg['loss']['lambda_ssim_input'] * loss_ssim +
                cfg['loss']['lambda_delta_l1'] * loss_delta_l1 +
                cfg['loss']['lambda_delta_tv'] * loss_delta_tv
            )

        scaler.scale(total).backward()
        scaler.step(opt_g)
        scaler.update()

        bs = kv.size(0)
        meter_total.update(total.item(), bs)
        meter_adv.update(loss_adv.item(), bs)
        meter_id.update(loss_id.item(), bs)
        meter_grad.update(loss_grad.item(), bs)
        meter_ssim.update(loss_ssim.item(), bs)
        meter_delta_l1.update(loss_delta_l1.item(), bs)
        meter_delta_tv.update(loss_delta_tv.item(), bs)
        meter_d.update(loss_d.item(), bs)

        if it % cfg['train']['print_iter_freq'] == 0 or it == len(loader):
            pbar.set_postfix({
                'loss': f'{meter_total.avg:.4f}',
                'adv': f'{meter_adv.avg:.4f}',
                'id': f'{meter_id.avg:.4f}',
                'grad': f'{meter_grad.avg:.4f}',
                'ssim': f'{meter_ssim.avg:.4f}',
                'd': f'{meter_d.avg:.4f}',
            })

    return {
        'loss': meter_total.avg,
        'adv': meter_adv.avg,
        'id': meter_id.avg,
        'grad': meter_grad.avg,
        'ssim': meter_ssim.avg,
        'delta_l1': meter_delta_l1.avg,
        'delta_tv': meter_delta_tv.avg,
        'd': meter_d.avg,
    }


@torch.no_grad()
def validate_and_save(model, loader, save_dir, device, cfg, epoch):
    model.eval()
    ensure_dir(save_dir)

    pbar = tqdm(loader, desc=f'Val {epoch:03d}')
    for idx, batch in enumerate(pbar):
        kv = batch['kv'].to(device)
        kv_path = batch['kv_path'][0]

        out = model(kv)
        pred = out['pred'][0, 0].detach().cpu().numpy()
        delta = out['delta'][0, 0].detach().cpu().numpy()
        kv_np = kv[0, 0].detach().cpu().numpy()

        ref_img = load_nii_image(kv_path)
        stem = Path(kv_path).name.replace('.nii.gz', '').replace('.nii', '')

        save_array_as_nii(pred, ref_img, os.path.join(save_dir, f'{stem}_epoch{epoch:03d}_pred.nii.gz'))
        save_array_as_nii(delta, ref_img, os.path.join(save_dir, f'{stem}_epoch{epoch:03d}_delta.nii.gz'))

        if cfg['val']['save_fused_pred']:
            fused = make_fused_pred_from_pred_input(
                pred, kv_np,
                sigma=cfg['val']['fused_sigma'],
                alpha=cfg['val']['fused_alpha'],
            )
            save_array_as_nii(fused, ref_img, os.path.join(save_dir, f'{stem}_epoch{epoch:03d}_fused_pred.nii.gz'))



def save_checkpoint(model, disc, opt_g, opt_d, scaler, epoch, save_path):
    ensure_dir(os.path.dirname(save_path))
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'disc': disc.state_dict(),
        'opt_g': opt_g.state_dict(),
        'opt_d': opt_d.state_dict(),
        'scaler': scaler.state_dict(),
    }, save_path)



def main():
    args = parse_args()
    print(f'[Info] Loading config from: {args.config}')
    cfg = load_yaml(args.config)

    seed_everything(cfg.get('seed', 1234))
    device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f'[Info] device = {device}')

    exp_dir = os.path.join(cfg['paths']['save_root'], cfg['exp_name'])
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    val_dir = os.path.join(exp_dir, 'val_outputs')
    ensure_dir(exp_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(val_dir)
    copy_file(args.config, os.path.join(exp_dir, 'used_config.yaml'))

    train_set = UnpairedNiiDataset(
        kv_dir=cfg['paths']['train_kv_dir'],
        drr_dir=cfg['paths']['train_drr_dir'],
        norm_mode=cfg['input']['norm_mode'],
        p_low=cfg['input']['p_low'],
        p_high=cfg['input']['p_high'],
        clip_min=cfg['input']['clip_min'],
        clip_max=cfg['input']['clip_max'],
        fixed_length=None,
    )
    val_set = ValNiiDataset(
        kv_dir=cfg['paths']['val_kv_dir'],
        drr_dir=cfg['paths']['val_drr_dir'],
        norm_mode=cfg['input']['norm_mode'],
        p_low=cfg['input']['p_low'],
        p_high=cfg['input']['p_high'],
        clip_min=cfg['input']['clip_min'],
        clip_max=cfg['input']['clip_max'],
        max_cases=cfg['val']['max_cases'],
    )

    print(f'[Info] train KV cases = {len(train_set.kv_paths)}')
    print(f'[Info] train DRR cases = {len(train_set.drr_paths)}')
    print(f'[Info] val KV cases   = {len(val_set.kv_paths)}')
    print(f'[Info] val used cases = {len(val_set)}')

    train_loader = DataLoader(
        train_set,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model, disc = build_model(cfg, device)
    print(f'[Model G] Params: {count_params(model):.3f} M')
    print(f'[Model D] Params: {count_params(disc):.3f} M')

    opt_g = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr_g'], betas=(cfg['train']['beta1'], cfg['train']['beta2']))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg['train']['lr_d'], betas=(cfg['train']['beta1'], cfg['train']['beta2']))
    scaler = GradScaler(enabled=cfg['train']['use_amp'])

    losses_obj = {
        'gan': GANLoss(cfg['loss']['gan_mode']).to(device),
        'grad': GradientLoss().to(device),
        'tv': TVLoss().to(device),
        'ssim': SSIMLoss(window_size=11).to(device),
    }

    for epoch in range(1, cfg['train']['epochs'] + 1):
        stats = train_one_epoch(model, disc, train_loader, opt_g, opt_d, scaler, losses_obj, device, cfg, epoch)
        print(
            f"[Epoch {epoch:03d}] loss={stats['loss']:.4f}, adv={stats['adv']:.4f}, id={stats['id']:.4f}, "
            f"grad={stats['grad']:.4f}, ssim={stats['ssim']:.4f}, d={stats['d']:.4f}"
        )

        latest_path = os.path.join(exp_dir, 'latest.pth')
        save_checkpoint(model, disc, opt_g, opt_d, scaler, epoch, latest_path)

        if epoch % cfg['train']['save_epoch_freq'] == 0:
            save_checkpoint(model, disc, opt_g, opt_d, scaler, epoch, os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth'))

        if epoch % cfg['train']['val_epoch_freq'] == 0:
            validate_and_save(model, val_loader, os.path.join(val_dir, f'epoch_{epoch:03d}'), device, cfg, epoch)


if __name__ == '__main__':
    main()
