import os
import cv2
import numpy as np
import SimpleITK as sitk


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_gray_image(path, img01):
    img = np.clip(img01, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    cv2.imwrite(path, img)


def tensor_to_numpy01(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.squeeze(x)
    return np.clip(x, 0.0, 1.0)


def save_numpy_as_nii_gz(np_img, out_path, reference_img=None, spacing=None, origin=None, direction=None):
    arr = np.asarray(np_img, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)

    if reference_img is not None:
        img.SetSpacing(reference_img.GetSpacing())
        img.SetOrigin(reference_img.GetOrigin())
        img.SetDirection(reference_img.GetDirection())
    else:
        if spacing is not None:
            img.SetSpacing(spacing)
        if origin is not None:
            img.SetOrigin(origin)
        if direction is not None:
            img.SetDirection(direction)

    sitk.WriteImage(img, out_path)