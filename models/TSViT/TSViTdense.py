"""
TSViT — Temporal-Spatial Vision Transformer for satellite time-series segmentation.

Architecture (Nyborg et al. 2022, "Seasonal Contrast"):
  1. Day-of-year temporal positional encoding (366-dim one-hot -> linear -> dim)
  2. Patch embedding: flatten spatial patches per time step
  3. Temporal transformer with N_classes learnable tokens PREPENDED
  4. Extract first N_classes tokens as class-discriminative summaries
  5. Spatial transformer across patches per class
  6. MLP head -> per-pixel class logits

Input tensor shape:  (B, T, H, W, C)  -- last dim is channels, last channel is DOY/365
Output tensor shape: (B, num_classes, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange

from models.TSViT.module import Attention, PreNorm, FeedForward


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TSViT(nn.Module):
    """
    Temporal-Spatial ViT for dense (per-pixel) land-cover segmentation.

    Parameters
    ----------
    model_config : dict
        img_res        : int   -- spatial size of input patch (pixels), e.g. 48
        patch_size     : int   -- ViT patch size (spatial), e.g. 1 (pixel-level)
        num_classes    : int   -- number of LC classes
        max_seq_len    : int   -- maximum number of time steps
        dim            : int   -- embedding dimension
        temporal_depth : int   -- transformer layers for temporal stage
        spatial_depth  : int   -- transformer layers for spatial stage
        heads          : int   -- attention heads
        dim_head       : int   -- dimension per head
        dropout        : float
        emb_dropout    : float
        pool           : str   -- 'cls' (kept for compat, unused)
        scale_dim      : int   -- MLP hidden = dim * scale_dim
        num_channels   : int   -- spectral channels INCLUDING the DOY channel
    """

    def __init__(self, model_config: dict):
        super().__init__()
        self.image_size     = model_config['img_res']
        self.patch_size     = model_config['patch_size']
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes    = model_config['num_classes']
        self.num_frames     = model_config['max_seq_len']
        self.dim            = model_config['dim']
        self.temporal_depth = model_config.get('temporal_depth', model_config['depth'])
        self.spatial_depth  = model_config.get('spatial_depth',  model_config['depth'])
        self.heads          = model_config['heads']
        self.dim_head       = model_config['dim_head']
        self.emb_dropout_p  = model_config['emb_dropout']
        self.scale_dim      = model_config['scale_dim']

        num_patches = self.num_patches_1d ** 2
        # num_channels - 1: exclude the DOY channel from spectral patch embedding
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2

        # Project flattened spatial patches to embedding dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)',
                p1=self.patch_size, p2=self.patch_size,
            ),
            nn.Linear(patch_dim, self.dim),
        )

        # Day-of-year positional encoding: one-hot(366) -> dim
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)

        # N_classes learnable class tokens prepended to temporal sequence
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))

        self.temporal_transformer = Transformer(
            self.dim, self.temporal_depth, self.heads, self.dim_head,
            self.dim * self.scale_dim, model_config['dropout'],
        )

        # Learnable spatial positional embedding (one per patch)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))

        self.space_transformer = Transformer(
            self.dim, self.spatial_depth, self.heads, self.dim_head,
            self.dim * self.scale_dim, model_config['dropout'],
        )

        self.dropout = nn.Dropout(self.emb_dropout_p)

        # Project each patch token to patch_size^2 pixel class scores
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size ** 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T, H, W, C)  -- last channel is DOY/365 in [0,1]
        returns : (B, num_classes, H, W)
        """
        x = x.permute(0, 1, 4, 2, 3)           # (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # --- Day-of-year temporal positional encoding ----------------------
        xt = x[:, :, -1, 0, 0]                  # (B, T)  DOY channel
        x  = x[:, :, :-1]                       # (B, T, C-1, H, W) spectral only

        xt = (xt * 365.0001).to(torch.int64).clamp(0, 365)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)   # (B, T, 366)
        xt = xt.reshape(-1, 366)
        temporal_pos = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)

        # --- Patch embedding -----------------------------------------------
        x = self.to_patch_embedding(x)          # (B*num_patches, T, dim)
        x = x.reshape(B, -1, T, self.dim)       # (B, num_patches, T, dim)
        x += temporal_pos.unsqueeze(1)           # broadcast over patches
        x = x.reshape(-1, T, self.dim)          # (B*num_patches, T, dim)

        # --- Temporal transformer (class tokens PREPENDED) -----------------
        cls = repeat(
            self.temporal_token, '() N d -> b N d',
            b=B * self.num_patches_1d ** 2,
        )
        x = torch.cat((cls, x), dim=1)          # (B*P, num_classes+T, dim)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]             # (B*P, num_classes, dim)

        # --- Spatial transformer -------------------------------------------
        x = (
            x.reshape(B, self.num_patches_1d ** 2, self.num_classes, self.dim)
             .permute(0, 2, 1, 3)
             .reshape(B * self.num_classes, self.num_patches_1d ** 2, self.dim)
        )
        x += self.space_pos_embedding
        x  = self.dropout(x)
        x  = self.space_transformer(x)

        # --- MLP head -> pixel logits --------------------------------------
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = (
            x.reshape(B, self.num_classes, self.num_patches_1d ** 2, self.patch_size ** 2)
             .permute(0, 2, 3, 1)
             .reshape(B, H, W, self.num_classes)
             .permute(0, 3, 1, 2)
        )
        return x   # (B, num_classes, H, W)


if __name__ == "__main__":
    cfg = {
        'img_res': 48, 'patch_size': 1, 'num_classes': 21,
        'max_seq_len': 24, 'dim': 128, 'temporal_depth': 4, 'spatial_depth': 2,
        'heads': 4, 'dim_head': 64, 'dropout': 0.1, 'emb_dropout': 0.1,
        'pool': 'cls', 'num_channels': 8, 'scale_dim': 4, 'depth': 4,
    }
    model = TSViT(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable parameters: {n_params:.2f}M")

    x = torch.rand(2, 24, 48, 48, 8)   # B=2, T=24, H=W=48, C=8 (7 spectral + 1 DOY)
    out = model(x)
    print(f"Output shape: {out.shape}")  # (2, 21, 48, 48)
