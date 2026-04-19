import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=True, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualToneMapper(nn.Module):
    """
    输出 delta，然后 pred = clamp(kv + delta, 0, 1)
    重点是：只改灰度，不鼓励重画结构。
    """
    def __init__(self, in_channels=1, base_channels=32, num_down=3, num_res_blocks=4,
                 delta_scale=0.2, use_tanh=True):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.use_tanh = bool(use_tanh)

        ch = base_channels
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, ch, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
        )

        downs = []
        enc_channels = [ch]
        for _ in range(num_down):
            downs.append(ConvBlock(ch, ch * 2, stride=2, norm=True, act=True))
            ch = ch * 2
            enc_channels.append(ch)
        self.downs = nn.Sequential(*downs)

        self.res_blocks = nn.Sequential(*[ResBlock(ch) for _ in range(num_res_blocks)])

        ups = []
        for _ in range(num_down):
            ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvBlock(ch, ch // 2, stride=1, norm=True, act=True),
            ))
            ch = ch // 2
        self.ups = nn.Sequential(*ups)

        self.tail = nn.Conv2d(ch, 1, 7, 1, 3)

    def forward(self, x):
        feat = self.head(x)
        feat = self.downs(feat)
        feat = self.res_blocks(feat)
        feat = self.ups(feat)
        delta = self.tail(feat)
        if self.use_tanh:
            delta = torch.tanh(delta)
        delta = delta * self.delta_scale
        pred = torch.clamp(x + delta, 0.0, 1.0)
        return {
            'pred': pred,
            'delta': delta,
        }


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels, base_channels, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            prev_nf_mult = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(base_channels * prev_nf_mult, base_channels * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(base_channels * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        prev_nf_mult = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(base_channels * prev_nf_mult, base_channels * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(base_channels * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(base_channels * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
