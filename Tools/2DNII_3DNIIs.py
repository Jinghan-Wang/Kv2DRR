import os
import glob
import re
import SimpleITK as sitk

def _natural_key(s: str):
    # 让 slice_2.nii.gz 排在 slice_10.nii.gz 前面
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(/d+)', os.path.basename(s))]

def stack_2d_nii_to_3d_nii(
    in_dir: str,
    out_path: str,
    pattern: str = "*.nii.gz",
    sort_mode: str = "natural",   # "natural" / "lex"
    ref_3d_path: str = None,      # 如果你有原始3D参考，传进来可完美复原spacing/origin/direction
):
    files = glob.glob(os.path.join(in_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(in_dir, pattern)}")

    if sort_mode == "natural":
        files.sort(key=_natural_key)
    else:
        files.sort()

    # 读取第一张，作为基准
    img0 = sitk.ReadImage(files[0])
    if img0.GetDimension() != 2:
        raise ValueError(f"Expect 2D slices, but got dim={img0.GetDimension()} for {files[0]}")

    size2d = img0.GetSize()        # (x,y)
    sp2d = img0.GetSpacing()       # (sx,sy)
    org2d = img0.GetOrigin()       # (ox,oy)
    dir2d = img0.GetDirection()    # (d00,d01,d10,d11)

    # 逐张读取并检查一致性
    slices = []
    for fp in files:
        im = sitk.ReadImage(fp)
        if im.GetDimension() != 2:
            raise ValueError(f"Not 2D: {fp}")
        if im.GetSize() != size2d:
            raise ValueError(f"Size mismatch: {fp}, got {im.GetSize()}, expect {size2d}")
        slices.append(im)

    # JoinSeries：把多张2D变成3D，z spacing 可以设置
    vol = sitk.JoinSeries(slices)

    if ref_3d_path is not None:
        # 有参考3D就直接复用（最准确）
        ref3d = sitk.ReadImage(ref_3d_path)
        if ref3d.GetDimension() != 3:
            raise ValueError("ref_3d_path must be a 3D image")
        vol.SetSpacing(ref3d.GetSpacing())
        vol.SetOrigin(ref3d.GetOrigin())
        vol.SetDirection(ref3d.GetDirection())
    else:
        # 没有参考3D：用2D信息 + z轴默认
        # z spacing：如果你切片文件名里有间隔信息就更好；这里先默认 1.0
        sx, sy = sp2d
        sz = 1.0

        vol.SetSpacing((sx, sy, sz))

        # 3D origin：用2D origin + z=0
        vol.SetOrigin((org2d[0], org2d[1], 0.0))

        # 3D direction：把2D方向嵌入到3D左上角，z轴设为(0,0,1)
        # [ d00 d01 0
        #   d10 d11 0
        #   0   0   1 ]
        d00, d01, d10, d11 = dir2d
        vol.SetDirection((
            d00, d01, 0.0,
            d10, d11, 0.0,
            0.0, 0.0, 1.0
        ))

    sitk.WriteImage(vol, out_path)
    print(f"Done. Stacked {len(files)} slices -> {out_path}")


if __name__ == "__main__":
    stack_2d_nii_to_3d_nii(
        in_dir="/data/WJH/03_SpinalKVEnhance/residual_tone_mapping/data/trainALL/pred",
        out_path="/data/WJH/03_SpinalKVEnhance/residual_tone_mapping/data/trainALL/20431pred.nii.gz",
        pattern="*.nii.gz",
        sort_mode="natural",
    )