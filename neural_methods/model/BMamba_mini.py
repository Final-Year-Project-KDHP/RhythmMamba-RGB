import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath
import math
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba

"""
Mini version of BMamba
Again using embed_dim=32, dim=8, etc.
"""

class Fusion_Stem_mini(nn.Module):
    def __init__(self, apha=0.5, belta=0.5, dim=8):
        super(Fusion_Stem_mini, self).__init__()
        self.stem11 = nn.Sequential(
            nn.Conv2d(1, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )
        self.stem12 = nn.Sequential(
            nn.Conv2d(4, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )
        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )
        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )
        self.apha = apha
        self.belta = belta

    def forward(self, x):
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:, :1], x[:, :1], x[:, :D - 2]], dim=1)
        x2 = torch.cat([x[:, :1], x[:, :D - 1]], dim=1)
        x3 = x
        x4 = torch.cat([x[:, 1:], x[:, D - 1:]], dim=1)
        x5 = torch.cat([x[:, 2:], x[:, D - 1:], x[:, D - 1:]], dim=1)

        x_diff_in = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2).view(N * D, 4, H, W)
        x_diff = self.stem12(x_diff_in)

        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        x_path1 = self.apha * x + self.belta * x_diff
        x_path1 = self.stem21(x_path1)
        x_path2 = self.stem22(x_diff)
        x = self.apha * x_path1 + self.belta * x_path2

        print("Fusion_Stem_mini (B) output shape:", x.shape)
        return x


class Attention_mask_mini(nn.Module):
    def forward(self, x):
        xsum = torch.sum(x, dim=(3, 4), keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5


class MambaLayer_mini(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(self.norm(x))


class Block_mamba_mini(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MambaLayer_mini(dim, d_state=16)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm1(x)))
        print("Block_mamba_mini (B) output shape:", x.shape)
        return x


class BMamba_mini(nn.Module):
    def __init__(self, depth=24, embed_dim=32, mlp_ratio=2, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.Fusion_Stem = Fusion_Stem_mini(dim=embed_dim // 4)
        self.attn_mask = Attention_mask_mini()

        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim // 4, embed_dim, kernel_size=(2, 5, 5),
                      stride=(2, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block_mamba_mini(embed_dim, mlp_ratio, drop_path=dpr[i]) 
            for i in range(depth)
        ])

        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, kernel_size=1, stride=1, padding=0)

        self.apply(partial(self._init_weights, n_layer=depth))
        self.apply(self._segm_init_weights)

    @staticmethod
    def _init_weights(module, n_layer, initializer_range=0.02):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _segm_init_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        B, D, C, H, W = x.shape
        print("Input Shape (B):", x.shape)

        x = self.Fusion_Stem(x)
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        print("After Fusion_Stem (B):", x.shape)

        x = self.stem3(x)
        print("After stem3 (B):", x.shape)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x, dim=(3, 4))
        print("After Mean Pooling (B):", x.shape)

        x = rearrange(x, 'b c t -> b t c')

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            print(f"After Block {i + 1} (B):", x.shape)

        x = self.upsample(x.permute(0, 2, 1))
        x = self.ConvBlockLast(x).squeeze(1)

        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
        print("Final Output Shape (B, normalized):", x.shape)
        return x
