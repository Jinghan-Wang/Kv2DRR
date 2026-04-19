import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def normalize_to_01(arr, mode='percentile', p_low=1.0, p_high=99.0, clip_min=None, clip_max=None):
    arr = arr.astype(np.float32)

    if clip_min is not None:
        arr = np.maximum(arr, float(clip_min))
    if clip_max is not None:
        arr = np.minimum(arr, float(clip_max))

    if mode == 'percentile':
        lo = np.percentile(arr, p_low)
        hi = np.percentile(arr, p_high)
    elif mode == 'minmax':
        lo = float(arr.min())
        hi = float(arr.max())
    else:
        raise ValueError(f'Unsupported norm_mode: {mode}')

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return arr.astype(np.float32)



def make_fused_pred_from_pred_input(pred_np, input_np, sigma=5.0, alpha=1.0):
    pred_np = pred_np.astype(np.float32)
    input_np = input_np.astype(np.float32)

    if cv2 is None:
        return pred_np

    pred_low = cv2.GaussianBlur(pred_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    input_low = cv2.GaussianBlur(input_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    input_high = input_np - input_low
    fused = pred_low + alpha * input_high
    fused = np.clip(fused, 0.0, 1.0)
    return fused.astype(np.float32)
