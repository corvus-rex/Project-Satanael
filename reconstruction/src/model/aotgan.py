import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .common import BaseNetwork


class InpaintGenerator(BaseNetwork):
    def __init__(self, args):
        super(InpaintGenerator, self).__init__()

        self.freeze_mode = getattr(args, "freeze_generator", None)

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(*[
            AOTBlock(256, args.rates) for _ in range(args.block_num)
        ])

        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True),
            UpConv(128, 64), nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)  # final head
        )

        self.init_weights()

        # Apply chosen freezing mode
        self.apply_freezing()

    def apply_freezing(self):
        if self.freeze_mode == "conf1":
            self.freeze_all_but_last_layer()

        elif self.freeze_mode == "conf2":
            self.freeze_encoder_and_middle()

    def freeze_all_but_last_layer(self):
        # Freeze everything
        for p in self.encoder.parameters(): p.requires_grad = False
        for p in self.middle.parameters(): p.requires_grad = False
        for p in self.decoder.parameters(): p.requires_grad = False

        # Unfreeze last layer
        last_layer = self.decoder[-1]  # Conv2d(64→3)
        for p in last_layer.parameters():
            p.requires_grad = True

    def freeze_encoder_and_middle(self):
        # Freeze encoder
        for p in self.encoder.parameters(): p.requires_grad = False
        # Freeze middle
        for p in self.middle.parameters(): p.requires_grad = False
        # Keep entire decoder trainable
        for p in self.decoder.parameters(): p.requires_grad = True


    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(
        self,
        args
    ):
        super(Discriminator, self).__init__()
        self.freeze_mode = getattr(args, "freeze_discriminator", None)
        
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

        self.init_weights()

        if self.freeze_mode:
            for p in self.netD.parameters():
                p.requires_grad = False
            if self.args.global_rank == 0:
                print("[**] Discriminator frozen: no D updates will occur.")

    def forward(self, x):
        feat = self.conv(x)
        return feat
