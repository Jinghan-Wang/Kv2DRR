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


def get_nii_stem(path):
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


def build_paired_paths(input_dir, target_dir):
    input_paths = list_nii_files(input_dir)
    target_paths = list_nii_files(target_dir)

    if len(input_paths) == 0:
        raise RuntimeError(f"No nii files found in {input_dir}")
    if len(target_paths) == 0:
        raise RuntimeError(f"No nii files found in {target_dir}")

    input_map = {}
    for path in input_paths:
        stem = get_nii_stem(path)
        if stem in input_map:
            raise RuntimeError(f"Duplicate input filename stem: {stem}")
        input_map[stem] = path

    target_map = {}
    for path in target_paths:
        stem = get_nii_stem(path)
        if stem in target_map:
            raise RuntimeError(f"Duplicate target filename stem: {stem}")
        target_map[stem] = path

    input_stems = set(input_map.keys())
    target_stems = set(target_map.keys())
    missing_targets = sorted(input_stems - target_stems)
    missing_inputs = sorted(target_stems - input_stems)

    if missing_targets or missing_inputs:
        errors = []
        if missing_targets:
            errors.append(f"missing targets for {missing_targets[:10]}")
        if missing_inputs:
            errors.append(f"missing inputs for {missing_inputs[:10]}")
        raise RuntimeError("Input/target filename mismatch: " + "; ".join(errors))

    return [(input_map[stem], target_map[stem]) for stem in sorted(input_stems)]


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
    return np.clip(x, 0.0, 1.0), vmin, vmax


def _norm_by_ref(x, ref_min, ref_max, eps=1e-6):
    x = (x - ref_min) / (ref_max - ref_min + eps)
    return np.clip(x, 0.0, 1.0)


def _build_inverted_channel(inp_raw, max_value):
    max_value = float(max_value)
    return np.clip(max_value - inp_raw, 0.0, max_value) / max(max_value, 1e-6)


def build_three_channel_input(inp_raw, inp_norm, aux_max_values=(1000.0, 2000.0)):
    if len(aux_max_values) != 2:
        raise ValueError(f"Expected 2 aux max values, got {aux_max_values}")

    aux_1000 = _build_inverted_channel(inp_raw, aux_max_values[0])
    aux_2000 = _build_inverted_channel(inp_raw, aux_max_values[1])
    stacked = np.stack([inp_norm, aux_1000, aux_2000], axis=0)
    return stacked.astype(np.float32), aux_1000.astype(np.float32), aux_2000.astype(np.float32)


def normalize_pair(inp, tgt, mode="pair_independent", fixed_range=(0.0, 4095.0), eps=1e-6):
    if mode == "pair_independent":
        inp_n, inp_min, inp_max = _norm_independent(inp, eps)
        tgt_n, tgt_min, tgt_max = _norm_independent(tgt, eps)
        meta = {
            "inp_min": inp_min,
            "inp_max": inp_max,
            "tgt_min": tgt_min,
            "tgt_max": tgt_max,
            "mode": mode,
        }
        return inp_n, tgt_n, meta

    if mode == "input_reference":
        ref_min = float(inp.min())
        ref_max = float(inp.max())
        inp_n = _norm_by_ref(inp, ref_min, ref_max, eps)
        tgt_n = _norm_by_ref(tgt, ref_min, ref_max, eps)
        meta = {
            "ref_min": ref_min,
            "ref_max": ref_max,
            "mode": mode,
        }
        return inp_n, tgt_n, meta

    if mode == "fixed_range":
        low, high = float(fixed_range[0]), float(fixed_range[1])
        inp_n = _norm_by_ref(np.clip(inp, low, high), low, high, eps)
        tgt_n = _norm_by_ref(np.clip(tgt, low, high), low, high, eps)
        meta = {
            "ref_min": low,
            "ref_max": high,
            "mode": mode,
        }
        return inp_n, tgt_n, meta

    raise ValueError(f"Unsupported normalize mode: {mode}")


class PairedNiiDataset(Dataset):
    def __init__(
        self,
        input_dir,
        target_dir,
        fixed_h=768,
        fixed_w=1088,
        normalize_mode="pair_independent",
        fixed_range=(0.0, 4095.0),
        aux_max_values=(1000.0, 2000.0),
    ):
        self.samples = build_paired_paths(input_dir, target_dir)
        self.fixed_h = fixed_h
        self.fixed_w = fixed_w
        self.normalize_mode = normalize_mode
        self.fixed_range = fixed_range
        self.aux_max_values = tuple(float(x) for x in aux_max_values)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        in_path, tg_path = self.samples[idx]

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

        inp_raw = inp_np.copy()
        inp_np, tgt_np, norm_meta = normalize_pair(
            inp_np,
            tgt_np,
            mode=self.normalize_mode,
            fixed_range=self.fixed_range,
        )

        inp_3ch, aux_1000_np, aux_2000_np = build_three_channel_input(
            inp_raw,
            inp_np,
            aux_max_values=self.aux_max_values,
        )

        inp = torch.from_numpy(inp_3ch).float()
        inp_base = torch.from_numpy(inp_np).unsqueeze(0).float()
        inp_aux_1000 = torch.from_numpy(aux_1000_np).unsqueeze(0).float()
        inp_aux_2000 = torch.from_numpy(aux_2000_np).unsqueeze(0).float()
        tgt = torch.from_numpy(tgt_np).unsqueeze(0).float()

        return {
            "input": inp,
            "input_base": inp_base,
            "input_aux_1000": inp_aux_1000,
            "input_aux_2000": inp_aux_2000,
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
