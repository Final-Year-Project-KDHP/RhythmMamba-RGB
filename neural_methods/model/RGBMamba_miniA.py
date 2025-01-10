import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# Import the three sub-model classes from your local paths:
from neural_methods.model.RMamba_mini import RMamba_mini
from neural_methods.model.GMamba_mini import GMamba_mini
from neural_methods.model.BMamba_mini import BMamba_mini


############################################################
# 1) Define the tri-stream cross-attention block
############################################################
class TriCrossAttentionBlock(nn.Module):
    """
    A single cross-attention block that processes three streams (R, G, B).
    Each stream attends to one of the others:
      - R -> attends to G
      - G -> attends to B
      - B -> attends to R

    Then each stream goes through an MLP (feed-forward) block.
    """
    def __init__(self, embed_dim=32, n_heads=4, ffn_dim=128, dropout=0.1):
        super().__init__()

        # PyTorch MultiheadAttention defaults to [L, B, E], but we can set batch_first=True
        self.cross_rg = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_gb = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_br = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        # LayerNorm for attention outputs
        self.ln_r1 = nn.LayerNorm(embed_dim)
        self.ln_g1 = nn.LayerNorm(embed_dim)
        self.ln_b1 = nn.LayerNorm(embed_dim)

        # Feed-forward layers (MLP) for each stream or a shared one:
        self.ffn_r = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_g = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # LayerNorm for feed-forward outputs
        self.ln_r2 = nn.LayerNorm(embed_dim)
        self.ln_g2 = nn.LayerNorm(embed_dim)
        self.ln_b2 = nn.LayerNorm(embed_dim)

    def forward(self, r, g, b):
        """
        r, g, b: [B, T, E] (batch_first=True).
        Returns updated (r_out, g_out, b_out).
        """
        # --- Cross-attention ---
        # R attends to G => Q=R, K=G, V=G
        r_attn, _ = self.cross_rg(query=r, key=g, value=g)
        # G attends to B => Q=G, K=B, V=B
        g_attn, _ = self.cross_gb(query=g, key=b, value=b)
        # B attends to R => Q=B, K=R, V=R
        b_attn, _ = self.cross_br(query=b, key=r, value=r)

        # Residual connection + LayerNorm
        r_out = self.ln_r1(r + r_attn)
        g_out = self.ln_g1(g + g_attn)
        b_out = self.ln_b1(b + b_attn)

        # --- Feed-forward ---
        r_ffn = self.ffn_r(r_out)
        g_ffn = self.ffn_g(g_out)
        b_ffn = self.ffn_b(b_out)

        # Residual connection + LayerNorm
        r_out = self.ln_r2(r_out + r_ffn)
        g_out = self.ln_g2(g_out + g_ffn)
        b_out = self.ln_b2(b_out + b_ffn)

        return r_out, g_out, b_out


############################################################
# 2) Define a tri-stream Transformer Encoder with 8 blocks
############################################################
class TriStreamTransformerEncoder(nn.Module):
    """
    Stacks multiple TriCrossAttentionBlocks to fuse the R/G/B streams.
    By default, we stack 8 blocks (the user requested about 8).
    Then we do a final fusion to produce a single channel timeseries.
    """
    def __init__(self,
                 embed_dim=32,
                 n_heads=4,
                 ffn_dim=128,
                 dropout=0.1,
                 num_layers=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TriCrossAttentionBlock(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final linear to merge the three streams => 1 channel
        # We'll do: concat(R, G, B) => linear => [B, T, 1]
        self.final_linear = nn.Linear(3 * embed_dim, 1)

    def forward(self, r, g, b):
        """
        r, g, b: [B, T, E]
        Return: [B, T, 1], the fused representation.
        """
        # Pass through multiple tri-cross-attn blocks
        for block in self.blocks:
            r, g, b = block(r, g, b)

        # Concatenate final R, G, B => [B, T, 3*E]
        fused_cat = torch.cat([r, g, b], dim=-1)
        # Linear down to 1 channel => [B, T, 1]
        fused = self.final_linear(fused_cat)
        return fused  # shape [B, T, 1]


############################################################
# 3) Define the Subnetwork Wrappers (R, G, B)
############################################################
class RMambaMiniFeatureExtractor(nn.Module):
    """
    A thin wrapper around the RMamba_mini that returns
    intermediate features of shape [B, 32, T].
    (We skip the final Conv1d.)
    """
    def __init__(self, depth=24, embed_dim=32, mlp_ratio=2, drop_path_rate=0.1):
        super().__init__()
        from .RMamba_mini import RMamba_mini
        model = RMamba_mini(depth=depth, embed_dim=embed_dim,
                            mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
        self.Fusion_Stem = model.Fusion_Stem
        self.stem3 = model.stem3
        self.attn_mask = model.attn_mask
        self.blocks = model.blocks
        self.upsample = model.upsample  # We'll use upsample at the end.

    def forward(self, x):
        """
        Input: x => [B, D, 1, H, W]
        Output: [B, 32, T'] (like original code)
        """
        B, D, C, H, W = x.shape
        print(f"[DEBUG] Entering RMambaMiniFeatureExtractor, x.shape = {x.shape}")

        x = self.Fusion_Stem(x)              # shape => [B, 8, D', H', W'] ...
        x = x.view(B, D, 8, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x, dim=(3, 4))        # [B, 32, T]
        x = rearrange(x, 'b c t -> b t c')   # => [B, T, 32]

        # Pass through Mamba blocks
        for blk in self.blocks:
            x = blk(x)

        # Now shape => [B, T, 32], but the original RMamba would end with upsample
        x = x.permute(0, 2, 1)               # => [B, 32, T]
        x = self.upsample(x)                 # => [B, 32, 2*T] if scale_factor=2
        return x


class GMambaMiniFeatureExtractor(nn.Module):
    def __init__(self, depth=24, embed_dim=32, mlp_ratio=2, drop_path_rate=0.1):
        super().__init__()
        from .GMamba_mini import GMamba_mini
        model = GMamba_mini(depth=depth, embed_dim=embed_dim,
                            mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
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
        x = self.upsample(x)
        return x


class BMambaMiniFeatureExtractor(nn.Module):
    def __init__(self, depth=24, embed_dim=32, mlp_ratio=2, drop_path_rate=0.1):
        super().__init__()
        from .BMamba_mini import BMamba_mini
        model = BMamba_mini(depth=depth, embed_dim=embed_dim,
                            mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
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
        x = self.upsample(x)
        return x


############################################################
# 4) Define the Main Model with the new TriStream Encoder
############################################################
class RGBMamba_miniA(nn.Module):
    """
    Combined mini-model that uses:
      - RMambaMiniFeatureExtractor on R => [B, 32, T']
      - GMambaMiniFeatureExtractor on G => [B, 32, T']
      - BMambaMiniFeatureExtractor on B => [B, 32, T']

    Then we do a tri-stream cross-attention with ~8 blocks (Transformer encoder).
    We end up with a single channel timeseries [B, T'] that is normalized.
    """
    def __init__(self,
                 depth=24,
                 embed_dim=32,
                 mlp_ratio=2,
                 drop_path_rate=0.1,
                 # Tri-stream encoder config:
                 cross_n_heads=4,
                 cross_ffn_dim=128,
                 cross_num_layers=8,
                 cross_dropout=0.1):
        super(RGBMamba_miniA, self).__init__()

        self.r_sub = RMambaMiniFeatureExtractor(depth=depth,
                                                embed_dim=embed_dim,
                                                mlp_ratio=mlp_ratio,
                                                drop_path_rate=drop_path_rate)
        self.g_sub = GMambaMiniFeatureExtractor(depth=depth,
                                                embed_dim=embed_dim,
                                                mlp_ratio=mlp_ratio,
                                                drop_path_rate=drop_path_rate)
        self.b_sub = BMambaMiniFeatureExtractor(depth=depth,
                                                embed_dim=embed_dim,
                                                mlp_ratio=mlp_ratio,
                                                drop_path_rate=drop_path_rate)

        # Tri-stream Transformer to fuse R/G/B
        self.fusion_encoder = TriStreamTransformerEncoder(
            embed_dim=embed_dim,      # dimension of each stream
            n_heads=cross_n_heads,
            ffn_dim=cross_ffn_dim,
            dropout=cross_dropout,
            num_layers=cross_num_layers
        )

    def forward(self, x):
        """
        x: [B, D, 3, H, W]
        We split into (R), (G), (B) => each [B, D, 1, H, W].
        Then subextractors => [B, 32, T'] each.
        Then we do tri-stream attention => final => [B, T'].
        """
        print(f"[DEBUG RGBMamba_miniA] input x.shape = {x.shape}")

        # Split channels
        x_r = x[:, :, 0:1, :, :]
        x_g = x[:, :, 1:2, :, :]
        x_b = x[:, :, 2:3, :, :]

        print(f"[DEBUG RGBMamba_miniA] x_r.shape = {x_r.shape}, x_g.shape = {x_g.shape}, x_b.shape = {x_b.shape}")

        # Extract features => [B, 32, T']
        feat_r = self.r_sub(x_r)  # => [B, 32, T']
        feat_g = self.g_sub(x_g)  # => [B, 32, T']
        feat_b = self.b_sub(x_b)  # => [B, 32, T']

        # Permute to [B, T', 32] for cross-attention
        feat_r = feat_r.permute(0, 2, 1)
        feat_g = feat_g.permute(0, 2, 1)
        feat_b = feat_b.permute(0, 2, 1)

        # Pass through the tri-stream cross-attention encoder => [B, T', 1]
        fused = self.fusion_encoder(feat_r, feat_g, feat_b)
        # fused => [B, T', 1]

        # Squeeze the channel dimension => [B, T']
        fused = fused.squeeze(-1)

        # Final normalization along time
        fused = (fused - fused.mean(dim=-1, keepdim=True)) / (fused.std(dim=-1, keepdim=True) + 1e-5)

        return fused
