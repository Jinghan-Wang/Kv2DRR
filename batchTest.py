import os
import glob
import cv2
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm

from models.residual_mapper import ResidualToneMapper
from datasets.paired_image_dataset import build_three_channel_input
from utils.io import ensure_dir, save_gray_image, save_numpy_as_nii_gz, tensor_to_numpy01

import setproctitle
setproctitle.setproctitle("WJH_Test")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# =========================
# 手动配置
# =========================
CKPT_PATH = r"/data/WJH/03_SpinalKVEnhance/07_KV2DRR/checkpoints/residual2399/epoch004.pth"
INPUT_DIR = r"/data/WJH/03_SpinalKVEnhance/07_KV2DRR/TestData/0-0489/kvNii"
PRED_DIR = r"/data/WJH/03_SpinalKVEnhance/07_KV2DRR/TestData/0-0489/pred"
NEW_PRED_DIR = r"/data/WJH/03_SpinalKVEnhance/07_KV2DRR/TestData/0-0489/SDRR"
DELTA_DIR = r"/data/WJH/03_SpinalKVEnhance/07_KV2DRR/TestData/0-0489/delta"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 和训练时保持一致
FIXED_H = 768
FIXED_W = 1088
NORMALIZE_MODE = "pair_independent"   # pair_independent | input_reference | fixed_range
FIXED_RANGE = (0.0, 4095.0)

# new_pred 后处理参数
NEW_PRED_SIGMA = 5.0
NEW_PRED_ALPHA = 1.0


# =========================
# 基础函数
# =========================
def read_nii_as_numpy(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)

    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected 2D nii or single-slice 3D nii, got shape={arr.shape}, path={path}")
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array after reading nii, got shape={arr.shape}, path={path}")

    return arr.astype(np.float32), img


def _norm_independent(x, eps=1e-6):
    vmin = float(x.min())
    vmax = float(x.max())
    x = (x - vmin) / (vmax - vmin + eps)
    x = np.clip(x, 0.0, 1.0)
    return x, vmin, vmax


def _norm_by_ref(x, ref_min, ref_max, eps=1e-6):
    x = (x - ref_min) / (ref_max - ref_min + eps)
    return np.clip(x, 0.0, 1.0)


def normalize_input(inp_np, mode="fixed_range", fixed_range=(0.0, 4095.0)):
    if mode == "pair_independent":
        inp_n, inp_min, inp_max = _norm_independent(inp_np)
        meta = {
            "mode": mode,
            "inp_min": inp_min,
            "inp_max": inp_max,
        }
        return inp_n, meta

    elif mode == "input_reference":
        ref_min = float(inp_np.min())
        ref_max = float(inp_np.max())
        inp_n = _norm_by_ref(inp_np, ref_min, ref_max)
        meta = {
            "mode": mode,
            "ref_min": ref_min,
            "ref_max": ref_max,
        }
        return inp_n, meta

    elif mode == "fixed_range":
        low, high = float(fixed_range[0]), float(fixed_range[1])
        inp_clip = np.clip(inp_np, low, high)
        inp_n = _norm_by_ref(inp_clip, low, high)
        meta = {
            "mode": mode,
            "ref_min": low,
            "ref_max": high,
        }
        return inp_n, meta

    else:
        raise ValueError(f"Unsupported normalize mode: {mode}")


def denormalize_output(pred_np, meta):
    mode = meta["mode"]

    if mode == "pair_independent":
        inp_min = meta["inp_min"]
        inp_max = meta["inp_max"]
        return pred_np * (inp_max - inp_min) + inp_min

    elif mode == "input_reference":
        ref_min = meta["ref_min"]
        ref_max = meta["ref_max"]
        return pred_np * (ref_max - ref_min) + ref_min

    elif mode == "fixed_range":
        ref_min = meta["ref_min"]
        ref_max = meta["ref_max"]
        return pred_np * (ref_max - ref_min) + ref_min

    else:
        raise ValueError(f"Unsupported normalize mode: {mode}")


def denormalize_like_train_validate(input_np, pred_np, new_pred_np, meta):
    mode = meta["mode"]

    pred_to_save = pred_np.astype(np.float32)
    new_pred_to_save = new_pred_np.astype(np.float32)
    input_to_save = input_np.astype(np.float32)

    if mode == "pair_independent":
        inp_min = float(meta["inp_min"])
        inp_max = float(meta["inp_max"])
        pred_to_save = pred_np * (inp_max - inp_min) + inp_min
        new_pred_to_save = new_pred_np * (4095.0 - 0.0) + 0.0
        input_to_save = input_np * (inp_max - inp_min) + inp_min
    elif mode == "input_reference":
        ref_min = float(meta["ref_min"])
        ref_max = float(meta["ref_max"])
        pred_to_save = pred_np * (ref_max - ref_min) + ref_min
        new_pred_to_save = new_pred_np * (ref_max - ref_min) + ref_min
        input_to_save = input_np * (ref_max - ref_min) + ref_min
    elif mode == "fixed_range":
        ref_min = float(meta["ref_min"])
        ref_max = float(meta["ref_max"])
        pred_to_save = pred_np * (ref_max - ref_min) + ref_min
        new_pred_to_save = new_pred_np * (ref_max - ref_min) + ref_min
        input_to_save = input_np * (ref_max - ref_min) + ref_min

    return input_to_save.astype(np.float32), pred_to_save.astype(np.float32), new_pred_to_save.astype(np.float32)


def make_new_pred_from_pred_input(pred_np, input_np, sigma=5.0, alpha=1.0):
    """
    pred_np, input_np: 2D numpy, range [0,1]
    new_pred = lowfreq(pred) + alpha * highfreq(input)
    """
    pred_np = pred_np.astype(np.float32)
    input_np = input_np.astype(np.float32)

    pred_low = cv2.GaussianBlur(pred_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    input_low = cv2.GaussianBlur(input_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    input_high = input_np - input_low

    new_pred = pred_low + alpha * input_high
    new_pred = np.clip(new_pred, 0.0, 1.0)
    return new_pred


# =========================
# 主流程
# =========================
@torch.no_grad()
def main():
    ensure_dir(PRED_DIR)
    ensure_dir(NEW_PRED_DIR)
    ensure_dir(DELTA_DIR)

    # 1) 读取 checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg = ckpt.get("cfg", {})
    dataset_cfg = cfg.get("dataset", {})

    model_cfg = cfg.get("model", {})
    in_channels = model_cfg.get("in_channels", 1)
    base_channels = model_cfg.get("base_channels", 16)
    delta_scale = model_cfg.get("delta_scale", 0.15)
    aux_max_values = tuple(dataset_cfg.get("aux_max_values", [1000.0, 2000.0]))
    fixed_h = int(dataset_cfg.get("fixed_h", FIXED_H))
    fixed_w = int(dataset_cfg.get("fixed_w", FIXED_W))
    normalize_mode = dataset_cfg.get("normalize_mode", NORMALIZE_MODE)
    fixed_range = tuple(dataset_cfg.get("fixed_range", FIXED_RANGE))

    model = ResidualToneMapper(
        in_channels=in_channels,
        base_channels=base_channels,
        delta_scale=delta_scale,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # 2) 收集输入文件
    nii_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.nii")) +
                       glob.glob(os.path.join(INPUT_DIR, "*.nii.gz")))

    if len(nii_files) == 0:
        raise RuntimeError(f"No nii/nii.gz files found in: {INPUT_DIR}")

    print(f"Found {len(nii_files)} files.")

    for nii_path in tqdm(nii_files, desc="Infer"):
        #if nii_path[:73][-5:] == "00316":
            inp_np_raw, ref_img = read_nii_as_numpy(nii_path)

            if inp_np_raw.shape != (fixed_h, fixed_w):
                print(f"[跳过] shape 不匹配: {nii_path}, got {inp_np_raw.shape}, expected {(FIXED_H, FIXED_W)}")
                continue

            # 3) 归一化
            inp_np_01, norm_meta = normalize_input(
                inp_np_raw,
                mode=normalize_mode,
                fixed_range=fixed_range,
            )

            # 4) 推理
            input_3ch, _, _ = build_three_channel_input(
                inp_np_raw,
                inp_np_01,
                aux_max_values=aux_max_values,
            )
            x = torch.from_numpy(input_3ch).unsqueeze(0).float().to(DEVICE)
            pred, delta = model(x)

            input_np = tensor_to_numpy01(x[0, :1])
            pred_np = tensor_to_numpy01(pred[0])

            delta_vis_np = tensor_to_numpy01(
                (delta[0] + delta_scale) / (2.0 * delta_scale + 1e-8)
            )

            new_pred_np = make_new_pred_from_pred_input(
                pred_np=pred_np,
                input_np=input_np,
                sigma=NEW_PRED_SIGMA,
                alpha=NEW_PRED_ALPHA,
            )

            # 5) 反归一化
            _, pred_raw, new_pred_raw = denormalize_like_train_validate(
                input_np=input_np,
                pred_np=pred_np,
                new_pred_np=new_pred_np,
                meta=norm_meta,
            )
            delta_vis_raw = delta_vis_np.astype(np.float32)

            # 6) 保存
            base = os.path.basename(nii_path)
            if base.endswith(".nii.gz"):
                stem = base[:-7]
            else:
                stem = os.path.splitext(base)[0]

            save_numpy_as_nii_gz(
                pred_raw,
                os.path.join(PRED_DIR, f"{stem}_pred.nii.gz"),
                reference_img=ref_img,
            )
            save_numpy_as_nii_gz(
                new_pred_raw,
                os.path.join(NEW_PRED_DIR, f"{stem}_new_pred.nii.gz"),
                reference_img=ref_img,
            )
            save_numpy_as_nii_gz(
                delta_vis_raw,
                os.path.join(DELTA_DIR, f"{stem}_delta_vis.nii.gz"),
                reference_img=ref_img,
            )

            #save_gray_image(os.path.join(PRED_DIR, f"{stem}_input.png"), input_np)
            #save_gray_image(os.path.join(PRED_DIR, f"{stem}_pred.png"), pred_np)
            #save_gray_image(os.path.join(PRED_DIR, f"{stem}_new_pred.png"), new_pred_np)
            #save_gray_image(os.path.join(PRED_DIR, f"{stem}_delta_vis.png"), delta_vis_np)

        #else:
        #    print(f"[失败] {nii_path}")

    print("全部测试完成。")


if __name__ == "__main__":
    main()
