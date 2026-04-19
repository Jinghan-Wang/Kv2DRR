import os
from glob import glob
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


NII_EXTS = ["*.nii", "*.nii.gz"]


def list_nii_files(folder):
    files = []
    for ext in NII_EXTS:
        files.extend(glob(os.path.join(folder, ext)))
    return sorted(files)


def read_nii_as_numpy(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)

    # 仅支持真正的 2D 或单层 3D
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
    return np.clip(x, 0.0, 1.0), vmin, vmax


def _norm_by_ref(x, ref_min, ref_max, eps=1e-6):
    x = (x - ref_min) / (ref_max - ref_min + eps)
    return np.clip(x, 0.0, 1.0)


def normalize_pair(inp, tgt, mode="pair_independent", fixed_range=(0.0, 4095.0), eps=1e-6):
    """
    pair_independent:
        inp/tgt 各自独立归一化到 [0,1]，最稳，适合先训练灰度风格映射。
    input_reference:
        用 input 的 min/max 同时归一化 inp/tgt，强调“以输入为参考系”。
    fixed_range:
        用固定强度窗归一化，适合已知稳定灰度范围的数据。
    """
    if mode == "pair_independent":
        inp_n, inp_min, inp_max = _norm_independent(inp, eps)
        tgt_n, tgt_min, tgt_max = _norm_independent(tgt, eps)
        meta = {
            "inp_min": inp_min, "inp_max": inp_max,
            "tgt_min": tgt_min, "tgt_max": tgt_max,
            "mode": mode,
        }
        return inp_n, tgt_n, meta

    if mode == "input_reference":
        ref_min = float(inp.min())
        ref_max = float(inp.max())
        inp_n = _norm_by_ref(inp, ref_min, ref_max, eps)
        tgt_n = _norm_by_ref(tgt, ref_min, ref_max, eps)
        meta = {
            "ref_min": ref_min, "ref_max": ref_max,
            "mode": mode,
        }
        return inp_n, tgt_n, meta

    if mode == "fixed_range":
        low, high = float(fixed_range[0]), float(fixed_range[1])
        inp_n = _norm_by_ref(np.clip(inp, low, high), low, high, eps)
        tgt_n = _norm_by_ref(np.clip(tgt, low, high), low, high, eps)
        meta = {
            "ref_min": low, "ref_max": high,
            "mode": mode,
        }
        return inp_n, tgt_n, meta

    raise ValueError(f"Unsupported normalize mode: {mode}")


class PairedNiiDataset(Dataset):
    def __init__(self, input_dir, target_dir, fixed_h=768, fixed_w=1088,
                 normalize_mode="pair_independent", fixed_range=(0.0, 4095.0)):
        self.input_paths = list_nii_files(input_dir)
        self.target_paths = list_nii_files(target_dir)
        self.fixed_h = fixed_h
        self.fixed_w = fixed_w
        self.normalize_mode = normalize_mode
        self.fixed_range = fixed_range

        if len(self.input_paths) == 0:
            raise RuntimeError(f"No nii files found in {input_dir}")
        if len(self.input_paths) != len(self.target_paths):
            raise RuntimeError(f"Input/target count mismatch: {len(self.input_paths)} vs {len(self.target_paths)}")

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.input_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.input_paths)}")
        in_path = self.input_paths[idx]
        tg_path = self.target_paths[idx]
        #print('in_path: ', in_path)
        #print('tg_path: ', tg_path)
        if os.path.basename(in_path) != os.path.basename(tg_path):
            print(f"[跳过] 文件名不一致: {os.path.basename(in_path)} vs {os.path.basename(tg_path)}")
            return self.__getitem__((idx + 1) % len(self.input_paths))

        inp_np, inp_img = read_nii_as_numpy(in_path)
        tgt_np, tgt_img = read_nii_as_numpy(tg_path)

        if inp_np.shape != tgt_np.shape:
            raise RuntimeError(
                f"Input/target shape mismatch: {inp_np.shape} vs {tgt_np.shape}"
                f"input={in_path}"
                f"target={tg_path}"
            )

        if inp_np.shape != (self.fixed_h, self.fixed_w):
            raise RuntimeError(
                f"Unexpected input shape {inp_np.shape}, expected {(self.fixed_h, self.fixed_w)}{in_path}"
            )
        if tgt_np.shape != (self.fixed_h, self.fixed_w):
            raise RuntimeError(
                f"Unexpected target shape {tgt_np.shape}, expected {(self.fixed_h, self.fixed_w)}{tg_path}"
            )

        inp_np, tgt_np, norm_meta = normalize_pair(
            inp_np, tgt_np,
            mode=self.normalize_mode,
            fixed_range=self.fixed_range,
        )

        inp = torch.from_numpy(inp_np).unsqueeze(0).float()   # [1,H,W]
        tgt = torch.from_numpy(tgt_np).unsqueeze(0).float()

        return {
            "input": inp,
            "target": tgt,
            "input_path": in_path,
            "target_path": tg_path,
            "norm_meta": norm_meta,
            "input_meta": {
                "spacing": inp_img.GetSpacing(),
                "origin": inp_img.GetOrigin(),
                "direction": inp_img.GetDirection(),
                "raw_shape": inp_np.shape,
            },
            "target_meta": {
                "spacing": tgt_img.GetSpacing(),
                "origin": tgt_img.GetOrigin(),
                "direction": tgt_img.GetDirection(),
                "raw_shape": tgt_np.shape,
            },
        }
