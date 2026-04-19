import os
import random
import shutil

import numpy as np
import torch
import yaml


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)



def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)



def save_yaml(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)



def copy_file(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
