import os
import yaml
import random
import numpy as np
import torch


def load_yaml(path):
    abs_path = os.path.abspath(path)
    print("Loading yaml from:", abs_path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print("YAML content preview:\n", text[:500])
    return yaml.safe_load(text)


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)