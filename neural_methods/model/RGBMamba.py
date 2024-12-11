# RGBMamba.py
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import math
from einops import rearrange
from timm.models.layers import trunc_normal_, lecun_normal_, DropPath
from mamba_ssm.modules.mamba_simple import Mamba
# Reuse Fusion_Stem, Attention_mask, Frequencydomain_FFN, MambaLayer, Block_mamba from RhythmMamba.py
# or import them if they are factored out into a separate file.

# Assume we can import them directly from RhythmMamba:
from neural_methods.model.RhythmMamba import Fusion_Stem, Attention_mask, Frequencydomain_FFN, Block_mamba, _init_weights, segm_init_weights

class RGBMambaBranch(nn.Module):
    """A single branch that processes one color channel video."""
    def __init__(self, depth=24, embed_dim=96, mlp_ratio=2, drop_path_rate=0.1, initializer_cfg=None, device=None, dtype=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

        # One-channel Fusion Stem (input: N,D,1,H,W)
        # Adjust Fusion_Stem to handle single-channel input:
        self.Fusion_Stem = Fusion_Stem(dim=embed_dim//4)
        self.attn_mask = Attention_mask()

        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim//4, embed_dim, kernel_size=(2, 5, 5), stride=(2, 1, 1),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.blocks = nn.ModuleList([Block_mamba(
            dim = embed_dim,
            mlp_ratio = mlp_ratio,
            drop_path=inter_dpr[i],
            norm_layer=nn.LayerNorm,
        ) for i in range(depth)])

        self.apply(segm_init_weights)
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, x):
        # x: [B, D, 1, H, W] for this channel
        B, D, C, H, W = x.shape
        x = self.Fusion_Stem(x)    #[B*D, C', H/8, W/8]
        x = x.view(B,D,self.embed_dim//4,H//8,W//8).permute(0,2,1,3,4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x,4)
        x = torch.mean(x,3)
        x = rearrange(x, 'b c t -> b t c')

        for blk in self.blocks:
            x = blk(x)
        # output: [B, T, C]
        return x

class RGBMamba(nn.Module):
    """RGBMamba model: processes R, G, B channels in parallel and then concatenates."""
    def __init__(self, 
                 depth=24, 
                 embed_dim=96, 
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()
        # Create three branches
        self.R_branch = RGBMambaBranch(depth, embed_dim, mlp_ratio, drop_path_rate, initializer_cfg, device, dtype, **kwargs)
        self.G_branch = RGBMambaBranch(depth, embed_dim, mlp_ratio, drop_path_rate, initializer_cfg, device, dtype, **kwargs)
        self.B_branch = RGBMambaBranch(depth, embed_dim, mlp_ratio, drop_path_rate, initializer_cfg, device, dtype, **kwargs)

        self.upsample = nn.Upsample(scale_factor=2)
        # After concatenation of three streams, dimension is 3 * embed_dim
        self.ConvBlockLast = nn.Conv1d(embed_dim*3, 1, kernel_size=1,stride=1, padding=0)

    def forward(self, x):
        # x: [B, D, C=3, H, W]
        R = x[:,:,[0],:,:]
        G = x[:,:,[1],:,:]
        B = x[:,:,[2],:,:]

        r_feat = self.R_branch(R) # [B, T, C]
        g_feat = self.G_branch(G)
        b_feat = self.B_branch(B)

        # Concatenate along feature dimension
        x_cat = torch.cat([r_feat, g_feat, b_feat], dim=2)  # [B,T, 3C]

        rPPG = x_cat.permute(0,2,1) # [B,3C,T]
        rPPG = self.upsample(rPPG)
        rPPG = self.ConvBlockLast(rPPG)    #[B, 1, D]
        rPPG = rPPG.squeeze(1)
        return rPPG
