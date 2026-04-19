import os
import random
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.nii_io import load_nii_as_array
from utils.image_ops import normalize_to_01


VALID_EXTS = ('.nii', '.nii.gz')


def _list_nii_files(folder):
    files = []
    for ext in VALID_EXTS:
        files.extend(glob(os.path.join(folder, f'*{ext}')))
    return sorted(files)


class UnpairedNiiDataset(Dataset):
    def __init__(self, kv_dir, drr_dir, norm_mode='percentile', p_low=1.0, p_high=99.0,
                 clip_min=None, clip_max=None, fixed_length=None):
        self.kv_paths = _list_nii_files(kv_dir)
        self.drr_paths = _list_nii_files(drr_dir)
        if len(self.kv_paths) == 0:
            raise RuntimeError(f'No KV files found in: {kv_dir}')
        if len(self.drr_paths) == 0:
            raise RuntimeError(f'No DRR files found in: {drr_dir}')

        self.norm_mode = norm_mode
        self.p_low = p_low
        self.p_high = p_high
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.fixed_length = fixed_length

    def __len__(self):
        if self.fixed_length is not None:
            return int(self.fixed_length)
        return max(len(self.kv_paths), len(self.drr_paths))

    def _load_one(self, path):
        arr, _ = load_nii_as_array(path)
        arr = normalize_to_01(
            arr,
            mode=self.norm_mode,
            p_low=self.p_low,
            p_high=self.p_high,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )
        arr = arr.astype(np.float32)
        arr = np.expand_dims(arr, axis=0)  # [1, H, W]
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        kv_path = self.kv_paths[idx % len(self.kv_paths)]
        drr_path = random.choice(self.drr_paths)

        kv = self._load_one(kv_path)
        drr = self._load_one(drr_path)

        return {
            'kv': kv,
            'drr': drr,
            'kv_path': kv_path,
            'drr_path': drr_path,
        }


class ValNiiDataset(Dataset):
    def __init__(self, kv_dir, drr_dir=None, norm_mode='percentile', p_low=1.0, p_high=99.0,
                 clip_min=None, clip_max=None, max_cases=None):
        self.kv_paths = _list_nii_files(kv_dir)
        if len(self.kv_paths) == 0:
            raise RuntimeError(f'No validation KV files found in: {kv_dir}')

        self.drr_paths = _list_nii_files(drr_dir) if drr_dir is not None and os.path.isdir(drr_dir) else []
        self.norm_mode = norm_mode
        self.p_low = p_low
        self.p_high = p_high
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.max_cases = max_cases

    def __len__(self):
        n = len(self.kv_paths)
        if self.max_cases is not None:
            n = min(n, int(self.max_cases))
        return n

    def _load_one(self, path):
        arr, _ = load_nii_as_array(path)
        arr = normalize_to_01(
            arr,
            mode=self.norm_mode,
            p_low=self.p_low,
            p_high=self.p_high,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )
        arr = arr.astype(np.float32)
        arr = np.expand_dims(arr, axis=0)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        kv_path = self.kv_paths[idx]
        kv = self._load_one(kv_path)

        sample = {
            'kv': kv,
            'kv_path': kv_path,
        }
        if len(self.drr_paths) > 0:
            drr_path = self.drr_paths[idx % len(self.drr_paths)]
            sample['drr'] = self._load_one(drr_path)
            sample['drr_path'] = drr_path
        return sample
