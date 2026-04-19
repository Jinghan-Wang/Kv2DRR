import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan'):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f'Unsupported gan_mode: {gan_mode}')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return torch.ones_like(prediction)
        return torch.zeros_like(prediction)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def _grad(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return gx, gy

    def forward(self, pred, target):
        gx1, gy1 = self._grad(pred)
        gx2, gy2 = self._grad(target)
        return F.l1_loss(gx1, gx2) + F.l1_loss(gy1, gy2)


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        loss_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        loss_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return loss_h + loss_w


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def _gaussian(self, window_size, sigma):
        gauss = torch.tensor([
            torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)))
            for x in range(window_size)
        ], dtype=torch.float32)
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel, device, dtype):
        _1d = self._gaussian(window_size, 1.5).to(device=device, dtype=dtype).unsqueeze(1)
        _2d = _1d @ _1d.t()
        window = _2d.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        c = img1.size(1)
        window = self._create_window(self.window_size, c, img1.device, img1.dtype)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=c)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=c) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=c) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
        )
        return 1.0 - ssim_map.mean()
