import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.TSViT.module import Attention, PreNorm, FeedForward, WindowAttention, window_partition, window_reverse
import numpy as np


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, window_size=window_size, 
            qkv_bias=True, attn_drop=dropout, proj_drop=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio*dim, dropout)
        
    def forward(self, x):
        # x shape: (B, L, C)
        B, L, C = x.shape
        
        H = W = int(L**0.5)
        
        # Window partitioning
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # Window attention
        
        x_windows = window_partition(shifted_x, self.window_size)
        B_win, Wh, Ww, C = x_windows.shape
        x_windows = x_windows.view(B_win, Wh * Ww, C)
        # Self-attention in windows
        
        attn_windows = self.attn(x_windows)
        
        attn_windows = attn_windows.view(B_win, Wh, Ww, C)
        
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, L, C)
        x = x + self.norm1(x)  # Residual connection for self-attention
        x = x + self.mlp(self.norm2(x))
        return x
    

class SwinTransformerBlock_dep(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, window_size=window_size, 
            qkv_bias=True, attn_drop=dropout, proj_drop=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio*dim, dropout)
        
    def forward(self, x):
        # x shape: (B, L, C)
        B, L, C = x.shape
        H = W = int(L**0.5)
        
        # Window partitioning
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # Window attention
        x_windows = window_partition(shifted_x, self.window_size)
        B_win, Wh, Ww, C = x_windows.shape
        x_windows = x_windows.view(B_win, Wh * Ww, C)
        attn_windows = self.attn(x_windows)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, L, C)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

class SegmentationHead(nn.Module):
  def __init__(self, input_sz, output_k):
    super(SegmentationHead, self).__init__()

    self.heads = nn.Sequential(
      nn.Conv2d(output_k, output_k, kernel_size=1,
                stride=1, dilation=1, padding=1, bias=False),
      nn.BatchNorm2d(output_k),
      nn.Softmax2d())

    self.input_sz = input_sz

  def forward(self, x):
    results = []
    x_i = self.heads(x)
    x_i = F.interpolate(x_i, size=self.input_sz, mode="bilinear")

    return results

class TSViTcls_m(nn.Module):
    """
    Temporal-Spatial ViT for object classification (used in main results, section 4.3)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.time_window = model_config['time_window']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        # self.depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(self.time_window+2, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.image_size**2)
            )
        self.segmentation_head = SegmentationHead(self.image_size, self.num_classes)

        
    def forward(self, x):
        #x = x.permute(0, 1, 4, 2, 3)

        B, T, C, H, W = x.shape
        
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * (150.+.0001)).to(torch.int64)
        
        xt = F.one_hot(xt, num_classes=self.time_window+2).to(torch.float32)
        xt = xt.reshape(-1, self.time_window+2)
        
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        
        x = self.to_patch_embedding(x)
        
        x = x.reshape(B, -1, T, self.dim)
        
        x += temporal_pos_embedding.unsqueeze(1)
        
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        
        x = self.temporal_transformer(x)
        
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        #cls_space_tokens = repeat(self.space_token, '() N d -> b N d', b=B * self.num_classes)
        #x = torch.cat((cls_space_tokens, x), dim=1)
        x = self.space_transformer(x)[:, 0]
        x = self.mlp_head(x.reshape(-1, self.dim))
        
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        x = self.segmentation_head(x)
        return x
    
    
class SegmentationHead(nn.Module):
  def __init__(self, input_sz, output_k):
    super(SegmentationHead, self).__init__()

    self.heads = nn.Sequential(
      nn.Conv2d(output_k, output_k, kernel_size=1,
                stride=1, dilation=1, padding=1, bias=False),
      nn.BatchNorm2d(output_k),
      nn.Softmax2d())

    self.input_sz = input_sz

  def forward(self, x):
    results = []
    x_i = self.heads(x)
    x_i = F.interpolate(x_i, size=self.input_sz, mode="bilinear")

    return x_i

class TSViTICC(nn.Module):
    """
    Temporal-Spatial ViT for object classification (used in main results, section 4.3)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.time_window = model_config['time_window']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        # self.depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(self.time_window+2, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.image_size**2)
            )
        self.segmentation_head = SegmentationHead(self.image_size, self.num_classes)
        
    def forward(self, x):
        #x = x.permute(0, 1, 4, 2, 3)

        B, T, C, H, W = x.shape
        
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * (150.+.0001)).to(torch.int64)
        
        xt = F.one_hot(xt, num_classes=self.time_window+2).to(torch.float32)
        xt = xt.reshape(-1, self.time_window+2)
        
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        
        x = self.to_patch_embedding(x)
        
        x = x.reshape(B, -1, T, self.dim)
        
        x += temporal_pos_embedding.unsqueeze(1)
        
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        
        x = self.temporal_transformer(x)
        
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        #cls_space_tokens = repeat(self.space_token, '() N d -> b N d', b=B * self.num_classes)
        #x = torch.cat((cls_space_tokens, x), dim=1)
        x = self.space_transformer(x)[:, 0]
        x = self.mlp_head(x.reshape(-1, self.dim))
        
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        x = self.segmentation_head(x)
        return x

class TSViT(nn.Module):
    """
    Temporal-Spatial ViT for object classification (used in main results, section 4.3)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.time_window = model_config['time_window']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        # self.depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(self.time_window+2, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
            )
        #self.segmentation_head = SegmentationHead(self.image_size, self.num_classes)
        
    def forward(self, x):
        #x = x.permute(0, 1, 4, 2, 3)

        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * (self.time_window-1+.0001)).to(torch.int64)
        xt = F.one_hot(xt, num_classes=self.time_window+2).to(torch.float32)
        xt = xt.reshape(-1, self.time_window+2)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)

        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)

        return x
    




class SwinTSViT(nn.Module):
    """
    Temporal-Spatial ViT for object classification (used in main results, section 4.3)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.time_window = model_config['time_window']
        self.temporal_depth = model_config.get('temporal_depth', model_config['depth'])
        self.spatial_depth = model_config.get('spatial_depth', model_config['depth'])
        # self.depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(self.time_window+2, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        #self.space_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        
        #self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        assert self.image_size % model_config.get('window_size', 4) == 0, "Window size must divide image size."
        self.space_transformer = nn.Sequential(*[
            SwinTransformerBlock(
                dim=self.dim,
                num_heads=self.heads,
                window_size=model_config.get('window_size', 4),
                shift_size=0 if (i % 2 == 0) else model_config.get('window_size', 4) // 2,
                mlp_ratio=model_config.get('mlp_ratio', 4),
                dropout=self.dropout
            ) for i in range(self.spatial_depth)
        ])
        self.dropout = nn.Dropout(self.emb_dropout)
        head_type = model_config.get('head', 'shallow')
        if head_type == 'shallow':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.patch_size**2)
                )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, 4*self.dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(4*self.dim, self.patch_size**2)
            )
                
        #self.segmentation_head = SegmentationHead(self.image_size, self.num_classes)
        
    def forward(self, x):
        #x = x.permute(0, 1, 4, 2, 3)

        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * (self.time_window-1+.0001)).to(torch.int64)
        xt = F.one_hot(xt, num_classes=self.time_window+2).to(torch.float32)
        xt = xt.reshape(-1, self.time_window+2)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)

        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))
        
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)

        return x


class TSViT_dino(nn.Module):
    """
    Temporal-Spatial ViT for object classification (used in main results, section 4.3)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.time_window = model_config['time_window']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        # self.depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(self.time_window+2, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
            )
        #self.segmentation_head = SegmentationHead(self.image_size, self.num_classes)
    
    def interpolate_pos_encoding(self, new_H, new_W):
        # Compute new number of patches
        new_num_patches_h = new_H // self.patch_size
        new_num_patches_w = new_W // self.patch_size
        new_num_patches = new_num_patches_h * new_num_patches_w
        
        orig_num_patches = self.space_pos_embedding.shape[1]
        if new_num_patches == orig_num_patches:
            return self.space_pos_embedding
        
        # Assume the original positional embeddings are on a square grid
        orig_grid_size = int(orig_num_patches ** 0.5)
        # Reshape to (1, dim, orig_grid_size, orig_grid_size)
        pos_embed = self.space_pos_embedding.reshape(1, orig_grid_size, orig_grid_size, self.dim).permute(0, 3, 1, 2)
        # Interpolate to new grid size
        pos_embed = F.interpolate(pos_embed, size=(new_num_patches_h, new_num_patches_w), 
                                mode='bilinear')
        # Reshape back to (1, new_num_patches, dim)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_num_patches, self.dim)
        return pos_embed
    
    def transformer_layers(self, x):
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * (self.time_window-1+.0001)).to(torch.int64)
        xt = F.one_hot(xt, num_classes=self.time_window+2).to(torch.float32)
        xt = xt.reshape(-1, self.time_window+2)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)

        x = self.to_patch_embedding(x)
        new_num_patches = (H // self.patch_size) * (W // self.patch_size)
        
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * (H // self.patch_size) * (W // self.patch_size))
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, new_num_patches, self.num_classes, self.dim).permute(0, 2, 1, 3)
        x = x.reshape(B*self.num_classes, new_num_patches, self.dim)
        
        #
        space_pos_embedding = self.interpolate_pos_encoding(H,W)
        x += space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)
        
        return x
    
    def reshape_to_singledim(self, x, b, mean = True):
        
        if mean:
            x = x.mean(dim=1)  # (B * num_classes, self.dim)
        else:
            x = x[:,0]
        # Now, reshape and aggregate over classes if needed. For instance,
        # if you use a [CLS] token for each crop, you might average them:
        
        x = x.reshape(b, self.num_classes, self.dim).mean(dim=1)  # (B, self.dim)
        return x
    
    def forward(self, x, oned_transform = True):
        #x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        
        x = self.transformer_layers(x)
        
        if oned_transform:
            x = self.reshape_to_singledim(x, B)

        # Return a global feature vector for each input crop.
        return x