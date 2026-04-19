import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps))


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0,  0,  0],
            [1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def get_grad(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return gx, gy

    def forward(self, pred, target):
        pred_gx, pred_gy = self.get_grad(pred)
        tgt_gx, tgt_gy = self.get_grad(target)
        return F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)


class DeltaRegularization(nn.Module):
    def forward(self, delta):
        return torch.mean(torch.abs(delta))


class DeltaTVLoss(nn.Module):
    """
    约束 delta 平滑，避免高频乱改小结构
    """
    def forward(self, delta):
        dx = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]).mean()
        dy = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :]).mean()
        return dx + dy


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def gaussian_window(self, channel, size, sigma=1.5):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
        return w.expand(channel, 1, size, size).contiguous()

    def forward(self, pred, target):
        c = pred.shape[1]
        w = self.gaussian_window(c, self.window_size).to(pred.device)

        mu1 = F.conv2d(pred, w, padding=self.window_size // 2, groups=c)
        mu2 = F.conv2d(target, w, padding=self.window_size // 2, groups=c)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, w, padding=self.window_size // 2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(target * target, w, padding=self.window_size // 2, groups=c) - mu2_sq
        sigma12 = F.conv2d(pred * target, w, padding=self.window_size // 2, groups=c) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
        )
        return 1.0 - ssim_map.mean()


class SoftHistogramLoss(nn.Module):
    def __init__(self, bins=64, sigma=0.02, value_range=(0.0, 1.0)):
        super().__init__()
        self.bins = bins
        self.sigma = sigma
        self.vmin, self.vmax = value_range
        centers = torch.linspace(self.vmin, self.vmax, bins)
        self.register_buffer("centers", centers)

    def _soft_hist(self, x):
        x = x.view(x.shape[0], -1, 1)   # [B,N,1]
        c = self.centers.view(1, 1, -1) # [1,1,bins]
        h = torch.exp(-0.5 * ((x - c) / self.sigma) ** 2)
        h = h.sum(dim=1)
        h = h / (h.sum(dim=1, keepdim=True) + 1e-8)
        return h

    def forward(self, pred, target):
        hp = self._soft_hist(pred)
        ht = self._soft_hist(target)
        return F.l1_loss(hp, ht)


class WeightedKeepLoss(nn.Module):
    """
    只在 input 的显著结构区域更强地约束 pred ~ input
    显著结构 = 梯度高 + 亮度高
    """
    def __init__(self, alpha_grad=2.0, beta_intensity=1.0, eps=1e-6):
        super().__init__()
        self.alpha_grad = alpha_grad
        self.beta_intensity = beta_intensity
        self.eps = eps

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0,  0,  0],
            [1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _normalize_map(self, x):
        # x: [B,1,H,W]
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + self.eps)

    def forward(self, pred, inp):
        gx = F.conv2d(inp, self.sobel_x, padding=1)
        gy = F.conv2d(inp, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + self.eps)

        grad_norm = self._normalize_map(grad_mag)
        inp_norm = self._normalize_map(inp)

        # 平坦区域权重接近 1，显著结构区域更大
        weight = 1.0 + self.alpha_grad * grad_norm + self.beta_intensity * inp_norm

        keep = torch.abs(pred - inp)
        return (weight * keep).mean()


class ToneMappingLoss(nn.Module):
    def __init__(self,
                 lambda_l1=1.0,
                 lambda_ssim=0.5,
                 lambda_grad=0.5,
                 lambda_delta=0.05,
                 lambda_hist=0.2,
                 lambda_grad_input=0.8,
                 lambda_keep=1.0,
                 lambda_ssim_input=0.2,
                 lambda_delta_tv=0.1):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.lambda_delta = lambda_delta
        self.lambda_hist = lambda_hist

        self.lambda_grad_input = lambda_grad_input
        self.lambda_keep = lambda_keep
        self.lambda_ssim_input = lambda_ssim_input
        self.lambda_delta_tv = lambda_delta_tv

        self.l1 = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()
        self.delta_reg = DeltaRegularization()
        self.delta_tv = DeltaTVLoss()
        self.hist = SoftHistogramLoss()
        self.keep = WeightedKeepLoss()

    def forward(self, pred, target, delta, inp):
        # -------- toward target --------
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_grad = self.grad(pred, target)
        loss_hist = self.hist(pred, target)

        # -------- keep input structure --------
        loss_grad_input = self.grad(pred, inp)
        loss_keep = self.keep(pred, inp)
        loss_ssim_input = self.ssim(pred, inp)

        # -------- constrain delta --------
        loss_delta = self.delta_reg(delta)
        loss_delta_tv = self.delta_tv(delta)

        total = (
            self.lambda_l1 * loss_l1 +
            self.lambda_ssim * loss_ssim +
            self.lambda_grad * loss_grad +
            self.lambda_hist * loss_hist +
            self.lambda_grad_input * loss_grad_input +
            self.lambda_keep * loss_keep +
            self.lambda_ssim_input * loss_ssim_input +
            self.lambda_delta * loss_delta +
            self.lambda_delta_tv * loss_delta_tv
        )

        logs = {
            "loss_total": total.item(),
            "loss_l1": loss_l1.item(),
            "loss_ssim": loss_ssim.item(),
            "loss_grad": loss_grad.item(),
            "loss_hist": loss_hist.item(),
            "loss_grad_input": loss_grad_input.item(),
            "loss_keep": loss_keep.item(),
            "loss_ssim_input": loss_ssim_input.item(),
            "loss_delta": loss_delta.item(),
            "loss_delta_tv": loss_delta_tv.item(),
        }
        return total, logs