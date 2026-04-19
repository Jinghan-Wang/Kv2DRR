import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calc_psnr(pred, target):
    return peak_signal_noise_ratio(target, pred, data_range=1.0)


def calc_ssim(pred, target):
    return structural_similarity(target, pred, data_range=1.0)