# KV → DRR 风格迁移工程（不配对 / 位置保持）

这是一个**最小可训练 PyTorch 工程**，目标是：

- 输入：KV（2D `nii.gz`）
- 输出：DRR-like 图像
- 训练方式：**不配对训练**（KV 和 DRR 不要求同名、不要求配准）
- 核心约束：**灰度/风格往 DRR 域靠近，但尽量保持 KV 的原始位置与结构**

## 1. 方法说明

网络不是直接“重画一张 DRR”，而是预测一个灰度残差：

```text
pred = clamp(kv + delta, 0, 1)
```

其中 `delta` 由网络输出。这样可以显著降低位置漂移风险。

### 训练损失

- `adv`：让输出分布更像 DRR
- `id`：输入本来就是 DRR 时，尽量保持不变
- `grad`：保持 KV 的边缘/结构
- `ssim_input`：保持 KV 的整体结构
- `delta_l1`：限制改动幅度
- `delta_tv`：让改动更平滑，减少伪影

## 2. 目录结构

```text
kv2drr_residual_unpaired_project/
├── configs/
│   └── default.yaml
├── datasets/
│   └── unpaired_nii_dataset.py
├── models/
│   └── networks.py
├── utils/
│   ├── image_ops.py
│   ├── losses.py
│   ├── misc.py
│   └── nii_io.py
├── train.py
├── infer.py
├── requirements.txt
└── README.md
```

## 3. 数据组织

```text
data/
├── train/
│   ├── kv/
│   │   ├── case001.nii.gz
│   │   ├── case002.nii.gz
│   │   └── ...
│   └── drr/
│       ├── drr001.nii.gz
│       ├── drr002.nii.gz
│       └── ...
└── val/
    ├── kv/
    │   ├── val001.nii.gz
    │   └── ...
    └── drr/
        ├── val_drr001.nii.gz
        └── ...
```

说明：

- 训练阶段 KV 和 DRR **不需要配对**。
- `val/drr` 主要用于身份损失和域观感参考；验证输出保存只依赖 `val/kv`。
- 当前工程假设每个 `nii.gz` 是 **2D 图像**；如果是 3D，但只有一个切片，也支持。

## 4. 安装依赖

```bash
pip install -r requirements.txt
```

## 5. 训练

先修改 `configs/default.yaml` 里的数据路径，然后运行：

```bash
python train.py --config configs/default.yaml
```

## 6. 推理

```bash
python infer.py \
  --ckpt outputs/exp_kv2drr/latest.pth \
  --input /path/to/test_kv_dir \
  --output /path/to/save_dir
```

也可以推理单个文件：

```bash
python infer.py \
  --ckpt outputs/exp_kv2drr/latest.pth \
  --input /path/to/case001.nii.gz \
  --output /path/to/save_dir
```

## 7. 推荐起步参数

如果你想先验证“能不能稳定收敛到 DRR 风格、同时不乱位置”，建议先用：

- `lambda_adv = 1.0`
- `lambda_id = 5.0`
- `lambda_grad = 2.0`
- `lambda_ssim_input = 2.0`
- `lambda_delta_l1 = 0.05`
- `lambda_delta_tv = 0.02`

如果输出仍然漂移：

- 提高 `lambda_grad`
- 提高 `lambda_ssim_input`
- 提高 `lambda_id`
- 减小 `delta_scale`

如果输出太像原始 KV、风格转不过去：

- 略微提高 `lambda_adv`
- 略微降低 `lambda_ssim_input`
- 略微增大 `delta_scale`

## 8. 当前工程的边界

这个工程适合你的任务起步，但它不是“物理真实 DRR 重建”。
它更适合做：

- KV 坐标保持
- DRR 域灰度/对比度迁移
- 为后续 spine-only 提供更稳定的域变换前处理

如果后面你要，我可以继续在这个工程上给你加：

1. 多窗宽输入（比如 3 通道 / 4 通道）
2. ROI / spine band 约束
3. 频域融合输出
4. 更强的轻量 U-Net 主干
5. 验证时自动保存中间 delta / pred / fused_pred
