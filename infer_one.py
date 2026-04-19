import os
from glob import glob
import numpy as np
import torch
from models.residual_mapper import ResidualToneMapper
from datasets.paired_image_dataset import read_nii_as_numpy, _norm_independent, _norm_by_ref
from utils.io import ensure_dir, save_gray_image, save_numpy_as_nii_gz, tensor_to_numpy01
from utils.misc import load_yaml


@torch.no_grad()
def main():
    cfg = load_yaml("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = cfg["infer"]["checkpoint"]
    input_dir = cfg["infer"]["input_dir"]
    output_dir = cfg["infer"]["output_dir"]
    ensure_dir(output_dir)

    model = ResidualToneMapper(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        delta_scale=cfg["model"]["delta_scale"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    files = sorted(glob(os.path.join(input_dir, "*.nii")) + glob(os.path.join(input_dir, "*.nii.gz")))

    norm_mode = cfg["dataset"]["normalize_mode"]
    fixed_range = tuple(cfg["dataset"]["fixed_range"])
    fixed_h = cfg["dataset"]["fixed_h"]
    fixed_w = cfg["dataset"]["fixed_w"]
    delta_scale = cfg["model"]["delta_scale"]

    for path in files:
        arr, ref_img = read_nii_as_numpy(path)
        if arr.shape != (fixed_h, fixed_w):
            raise RuntimeError(f"Unexpected input shape {arr.shape}, expected {(fixed_h, fixed_w)}: {path}")

        if norm_mode == "pair_independent":
            arr01, in_min, in_max = _norm_independent(arr)
            denorm_min, denorm_max = in_min, in_max
        elif norm_mode == "input_reference":
            ref_min = float(arr.min())
            ref_max = float(arr.max())
            arr01 = _norm_by_ref(arr, ref_min, ref_max)
            denorm_min, denorm_max = ref_min, ref_max
        elif norm_mode == "fixed_range":
            low, high = float(fixed_range[0]), float(fixed_range[1])
            arr01 = _norm_by_ref(np.clip(arr, low, high), low, high)
            denorm_min, denorm_max = low, high
        else:
            raise ValueError(f"Unsupported normalize mode: {norm_mode}")

        x = torch.from_numpy(arr01).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,768,1088]
        pred, delta = model(x)

        pred01 = tensor_to_numpy01(pred[0])
        delta_vis = tensor_to_numpy01((delta[0] + delta_scale) / (2 * delta_scale + 1e-8))
        pred_raw = pred01 * (denorm_max - denorm_min) + denorm_min

        base = os.path.basename(path)
        stem = base[:-7] if base.endswith(".nii.gz") else os.path.splitext(base)[0]

        save_numpy_as_nii_gz(pred_raw, os.path.join(output_dir, f"{stem}_pred.nii.gz"), reference_img=ref_img)
        save_gray_image(os.path.join(output_dir, f"{stem}_pred.png"), pred01)
        save_gray_image(os.path.join(output_dir, f"{stem}_delta_vis.png"), delta_vis)

    print("Inference done.")


if __name__ == "__main__":
    main()