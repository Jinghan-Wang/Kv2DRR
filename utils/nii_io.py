import os

import numpy as np
import SimpleITK as sitk


def load_nii_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sitk.ReadImage(path)



def load_nii_as_array(path):
    img = load_nii_image(path)
    arr = sitk.GetArrayFromImage(img)

    # 2D: [H, W]
    # 3D: [D, H, W]
    if arr.ndim == 2:
        arr2d = arr
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr2d = arr[0]
        else:
            raise ValueError(
                f'Only 2D nii.gz (or 3D with D=1) is supported. Got shape={arr.shape} for {path}'
            )
    else:
        raise ValueError(f'Unsupported array ndim={arr.ndim} for {path}')

    return arr2d.astype(np.float32), img



def save_array_as_nii(arr2d, ref_img, save_path):
    arr2d = np.asarray(arr2d, dtype=np.float32)
    out = sitk.GetImageFromArray(arr2d)

    # ref_img 可以是 2D，也可以是 D=1 的 3D
    if ref_img.GetDimension() == 2:
        out.SetSpacing(ref_img.GetSpacing())
        out.SetOrigin(ref_img.GetOrigin())
        out.SetDirection(ref_img.GetDirection())
    elif ref_img.GetDimension() == 3:
        spacing = ref_img.GetSpacing()
        origin = ref_img.GetOrigin()
        direction = ref_img.GetDirection()
        out.SetSpacing((spacing[1], spacing[2]))
        out.SetOrigin((origin[1], origin[2]))
        out.SetDirection((direction[4], direction[5], direction[7], direction[8]))
    else:
        raise ValueError(f'Unsupported ref_img dimension: {ref_img.GetDimension()}')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sitk.WriteImage(out, save_path)
