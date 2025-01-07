import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# Import the three sub-model classes
# (Make sure these imports match your actual local paths!)
from neural_methods.model.RMamba import RMamba
from neural_methods.model.GMamba import GMamba
from neural_methods.model.BMamba import BMamba


class FusionBlock(nn.Module):
    """
    Example fusion block:
      - Takes three features each of shape [B, 96, T]
      - Concatenates along channel dim -> [B, 288, T]
      - 1D conv layers to reduce to [B, 1, T]
      - Could add normalization, activation, etc.
    """
    def __init__(self, in_channels=288, hidden_channels=96, out_channels=1):
        super(FusionBlock, self).__init__()

        # Example architecture: (Conv1d -> ReLU -> Conv1d)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        x: [B, in_channels=288, T]
        return: [B, 1, T]
        """
        x = self.conv1(x)          # => [B, hidden_channels, T]
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)         # => [B, out_channels=1, T]
        x = self.bn2(x)

        return x


class RMambaFeatureExtractor(RMamba):
    """
    A thin wrapper around RMamba that returns intermediate
    features of shape [B, 96, T] (before the final Conv1d).
    """

    def forward(self, x):
        """
        We'll copy over the original forward() from RMamba but
        STOP before the final ConvBlockLast.

        Output shape should be [B, embed_dim, T] = [B, 96, 2*(D//2)]
        (Typically T is something like 160.)
        """
        print(f"[DEBUG] Entering RMambaFeatureExtractor, x.shape = {x.shape}")
        

        B, D, C, H, W = x.shape

        print(f"[DEBUG RMambaFeatureExtractor] B={B},D={D},C={C},H={H},W={W}")


        # 1) Fusion Stem
        x = self.Fusion_Stem(x)
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)

        # 2) 3D Convolution
        x = self.stem3(x)

        # 3) Attention Mask
        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        # 4) Mean Pool across spatial dims
        x = torch.mean(x, dim=(3, 4))

        # 5) Rearrange to [B, T, C]
        x = rearrange(x, 'b c t -> b t c')

        # 6) Pass through each Mamba block
        for blk in self.blocks:
            x = blk(x)

        # 7) Upsample => shape [B, embed_dim, 2*T]
        #    Instead of final conv, we just return x as [B, embed_dim, T'].
        #    So:
        x = x.permute(0, 2, 1)  # => [B, C=embed_dim, T]
        x = self.upsample(x)    # => [B, 96, 2T or 2*(D//2)]
        return x


class GMambaFeatureExtractor(GMamba):
    """
    Same idea for GMamba: returns features of shape [B, 96, T]
    before final Conv1d.
    """
    def forward(self, x):
        print(f"[DEBUG] Entering GMambaFeatureExtractor, x.shape = {x.shape}")


        B, D, C, H, W = x.shape

        print(f"[DEBUG GMambaFeatureExtractor] B={B},D={D},C={C},H={H},W={W}")

        x = self.Fusion_Stem(x)
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x, dim=(3, 4))
        x = rearrange(x, 'b c t -> b t c')

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        return x


class BMambaFeatureExtractor(BMamba):
    """
    Same idea for BMamba: returns features of shape [B, 96, T]
    before final Conv1d.
    """
    def forward(self, x):

        print(f"[DEBUG] Entering BMambaFeatureExtractor, x.shape = {x.shape}")

        B, D, C, H, W = x.shape

        print(f"[DEBUG BMambaFeatureExtractor] B={B},D={D},C={C},H={H},W={W}")


        x = self.Fusion_Stem(x)
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x, dim=(3, 4))
        x = rearrange(x, 'b c t -> b t c')

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        return x


class RGBMamba(nn.Module):
    """
    Combined model that uses:
      - RMambaFeatureExtractor on the R channel
      - GMambaFeatureExtractor on the G channel
      - BMambaFeatureExtractor on the B channel

    Then fuses the three [B, 96, T] feature maps with a FusionBlock
    to produce [B, 1, T], finally squeezes to [B, T] and normalizes.
    """
    def __init__(self, depth=24, embed_dim=96, mlp_ratio=2, drop_path_rate=0.1):
        super(RGBMamba, self).__init__()

        # Submodels
        self.r_sub = RMambaFeatureExtractor(
            depth=depth,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )
        self.g_sub = GMambaFeatureExtractor(
            depth=depth,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )
        self.b_sub = BMambaFeatureExtractor(
            depth=depth,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )

        # A simple fusion block
        # Each submodel outputs [B, 96, T] => fused => [B, 1, T].
        self.fusion = FusionBlock(in_channels=embed_dim * 3, hidden_channels=embed_dim, out_channels=1)

    def forward(self, x):
        """
        x: [B, D, 3, H, W]
        Output: [B, T] after final normalization
        """

        print(f"[DEBUG RGBMamba] input x.shape = {x.shape}")

        # Split channels for each submodel
        x_r = x[:, :, 0:1, :, :]  # Red
        x_g = x[:, :, 1:2, :, :]  # Green
        x_b = x[:, :, 2:3, :, :]  # Blue

        print(f"[DEBUG RGBMamba] x_r.shape = {x_r.shape}, x_g.shape = {x_g.shape}, x_b.shape = {x_b.shape}")

        # Get feature maps: each [B, 96, T]
        feat_r = self.r_sub(x_r)
        feat_g = self.g_sub(x_g)
        feat_b = self.b_sub(x_b)

        # Concatenate along channel dim => [B, 288, T]
        feat_cat = torch.cat([feat_r, feat_g, feat_b], dim=1)

        # Fuse => [B, 1, T]
        fused = self.fusion(feat_cat)

        # Squeeze => [B, T]
        fused = fused.squeeze(1)

        # Final normalization => [B, T]
        fused = (fused - fused.mean(dim=-1, keepdim=True)) / (fused.std(dim=-1, keepdim=True) + 1e-5)

        return fused
