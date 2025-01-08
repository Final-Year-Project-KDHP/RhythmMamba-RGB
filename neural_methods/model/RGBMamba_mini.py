import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# Import the three sub-model classes from your local paths:
from neural_methods.model.RMamba_mini import RMamba_mini
from neural_methods.model.GMamba_mini import GMamba_mini
from neural_methods.model.BMamba_mini import BMamba_mini

"""
Mini version of RGBMamba:
 - Each submodel now produces [B, 32, T] (instead of [B, 96, T])
 - The fusion block merges them => [B, 96, T], then we reduce to 1 channel
"""

class FusionBlockMini(nn.Module):
    """
    Fusion block for the mini version:
      - Takes three features each of shape [B, 32, T]
      - Concatenates => [B, 96, T]
      - 1D conv layers to reduce to [B, 1, T]
    """
    def __init__(self, in_channels=96, hidden_channels=32, out_channels=1):
        super(FusionBlockMini, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        x: [B, in_channels=96, T]
        return: [B, 1, T]
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x


class RMambaMiniFeatureExtractor(nn.Module):
    """
    A thin wrapper around the RMamba_mini that returns
    intermediate features of shape [B, 32, T].
    (We skip the final Conv1d.)
    """
    def __init__(self, depth=24, embed_dim=32, mlp_ratio=2, drop_path_rate=0.1):
        super().__init__()
        # You can copy the logic from RMamba_mini but cut it short before final.
        from .RMamba_mini import RMamba_mini, Block_mamba_mini, Fusion_Stem_mini, MambaLayer_mini
        model = RMamba_mini(depth=depth, embed_dim=embed_dim, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
        self.Fusion_Stem = model.Fusion_Stem
        self.stem3 = model.stem3
        self.attn_mask = model.attn_mask
        self.blocks = model.blocks
        self.upsample = model.upsample  # We'll use upsample at the end.

    def forward(self, x):
        B, D, C, H, W = x.shape
        print(f"[DEBUG] Entering RMambaMiniFeatureExtractor, x.shape = {x.shape}")

        x = self.Fusion_Stem(x)
        x = x.view(B, D, 8, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x, dim=(3, 4))
        x = rearrange(x, 'b c t -> b t c')

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(0, 2, 1)
        x = self.upsample(x)  # => shape [B, 32, 2*T]
        return x


class GMambaMiniFeatureExtractor(nn.Module):
    """Similar logic for GMamba_mini"""
    def __init__(self, depth=24, embed_dim=32, mlp_ratio=2, drop_path_rate=0.1):
        super().__init__()
        from .GMamba_mini import GMamba_mini
        model = GMamba_mini(depth=depth, embed_dim=embed_dim, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
        self.Fusion_Stem = model.Fusion_Stem
        self.stem3 = model.stem3
        self.attn_mask = model.attn_mask
        self.blocks = model.blocks
        self.upsample = model.upsample

    def forward(self, x):
        B, D, C, H, W = x.shape
        print(f"[DEBUG] Entering GMambaMiniFeatureExtractor, x.shape = {x.shape}")

        x = self.Fusion_Stem(x)
        x = x.view(B, D, 8, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x, dim=(3, 4))
        x = rearrange(x, 'b c t -> b t c')

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(0, 2, 1)
        x = self.upsample(x)  # => [B, 32, 2*T]
        return x


class BMambaMiniFeatureExtractor(nn.Module):
    """Similar logic for BMamba_mini"""
    def __init__(self, depth=24, embed_dim=32, mlp_ratio=2, drop_path_rate=0.1):
        super().__init__()
        from .BMamba_mini import BMamba_mini
        model = BMamba_mini(depth=depth, embed_dim=embed_dim, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
        self.Fusion_Stem = model.Fusion_Stem
        self.stem3 = model.stem3
        self.attn_mask = model.attn_mask
        self.blocks = model.blocks
        self.upsample = model.upsample

    def forward(self, x):
        B, D, C, H, W = x.shape
        print(f"[DEBUG] Entering BMambaMiniFeatureExtractor, x.shape = {x.shape}")

        x = self.Fusion_Stem(x)
        x = x.view(B, D, 8, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x, dim=(3, 4))
        x = rearrange(x, 'b c t -> b t c')

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(0, 2, 1)
        x = self.upsample(x)  # => [B, 32, 2*T]
        return x


class RGBMamba_mini(nn.Module):
    """
    Combined mini-model that uses:
      - RMambaMiniFeatureExtractor on the R channel => [B, 32, T']
      - GMambaMiniFeatureExtractor on the G channel => [B, 32, T']
      - BMambaMiniFeatureExtractor on the B channel => [B, 32, T']
    Then fuses the three [B, 32, T'] => [B, 96, T'], => [B, 1, T'] => [B, T']
    """
    def __init__(self, depth=24, embed_dim=32, mlp_ratio=2, drop_path_rate=0.1):
        super(RGBMamba_mini, self).__init__()

        self.r_sub = RMambaMiniFeatureExtractor(
            depth=depth,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )
        self.g_sub = GMambaMiniFeatureExtractor(
            depth=depth,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )
        self.b_sub = BMambaMiniFeatureExtractor(
            depth=depth,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )

        # After we have 3 * 32 => 96 channels
        self.fusion = FusionBlockMini(in_channels=96, hidden_channels=32, out_channels=1)

    def forward(self, x):
        """
        x: [B, D, 3, H, W]
        We split into (R), (G), (B) => each [B, D, 1, H, W]
        """
        print(f"[DEBUG RGBMamba_mini] input x.shape = {x.shape}")

        x_r = x[:, :, 0:1, :, :]
        x_g = x[:, :, 1:2, :, :]
        x_b = x[:, :, 2:3, :, :]

        print(f"[DEBUG RGBMamba_mini] x_r.shape = {x_r.shape}, x_g.shape = {x_g.shape}, x_b.shape = {x_b.shape}")

        feat_r = self.r_sub(x_r)  # => [B, 32, T']
        feat_g = self.g_sub(x_g)  # => [B, 32, T']
        feat_b = self.b_sub(x_b)  # => [B, 32, T']

        feat_cat = torch.cat([feat_r, feat_g, feat_b], dim=1)  # => [B, 96, T']
        fused = self.fusion(feat_cat)  # => [B, 1, T']

        fused = fused.squeeze(1)       # => [B, T']

        # Final normalization
        fused = (fused - fused.mean(dim=-1, keepdim=True)) / (fused.std(dim=-1, keepdim=True) + 1e-5)
        return fused
