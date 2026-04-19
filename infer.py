import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch

from models.networks import ResidualToneMapper
from utils.image_ops import normalize_to_01, make_fused_pred_from_pred_input
from utils.nii_io import load_nii_as_array, load_nii_image, save_array_as_nii



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--input', type=str, required=True, help='single nii.gz file or directory')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--num_down', type=int, default=3)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    parser.add_argument('--delta_scale', type=float, default=0.20)
    parser.add_argument('--use_tanh', action='store_true', default=True)
    parser.add_argument('--norm_mode', type=str, default='percentile')
    parser.add_argument('--p_low', type=float, default=1.0)
    parser.add_argument('--p_high', type=float, default=99.0)
    parser.add_argument('--save_fused_pred', action='store_true')
    parser.add_argument('--fused_sigma', type=float, default=5.0)
    parser.add_argument('--fused_alpha', type=float, default=1.0)
    return parser.parse_args()



def list_inputs(path):
    if os.path.isfile(path):
        return [path]
    files = []
    files.extend(glob(os.path.join(path, '*.nii')))
    files.extend(glob(os.path.join(path, '*.nii.gz')))
    return sorted(files)



def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = ResidualToneMapper(
        in_channels=1,
        base_channels=args.base_channels,
        num_down=args.num_down,
        num_res_blocks=args.num_res_blocks,
        delta_scale=args.delta_scale,
        use_tanh=args.use_tanh,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    paths = list_inputs(args.input)
    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        for p in paths:
            arr, _ = load_nii_as_array(p)
            arr01 = normalize_to_01(arr, mode=args.norm_mode, p_low=args.p_low, p_high=args.p_high)
            x = torch.from_numpy(arr01[None, None].astype(np.float32)).to(device)
            out = model(x)
            pred = out['pred'][0, 0].cpu().numpy()
            delta = out['delta'][0, 0].cpu().numpy()

            ref_img = load_nii_image(p)
            stem = Path(p).name.replace('.nii.gz', '').replace('.nii', '')
            save_array_as_nii(pred, ref_img, os.path.join(args.output, f'{stem}_pred.nii.gz'))
            save_array_as_nii(delta, ref_img, os.path.join(args.output, f'{stem}_delta.nii.gz'))

            if args.save_fused_pred:
                fused = make_fused_pred_from_pred_input(pred, arr01, sigma=args.fused_sigma, alpha=args.fused_alpha)
                save_array_as_nii(fused, ref_img, os.path.join(args.output, f'{stem}_fused_pred.nii.gz'))

            print(f'[Saved] {stem}')


if __name__ == '__main__':
    main()
