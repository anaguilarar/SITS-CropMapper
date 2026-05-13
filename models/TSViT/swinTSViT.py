import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange
from .module import Attention, PreNorm, Attention
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_
from functools import partial

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
    
def window_partition(x, window_size):
    """Partitions input into non-overlapping windows.
    
    Args:
        x: (B, H, W, C)
        window_size: int
    
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, 
        H // window_size, window_size,
        W // window_size, window_size,
        C
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H//window_size, W//window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class FeedForward(nn.Module):
    """Standard MLP block with GELU activation."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape  # (Batch, Time, Dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, N, self.heads, -1).permute(0, 2, 1, 3) for t in qkv]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            mask = mask[:, None, None, :].to(attn.dtype)  # (B, 1, 1, Time)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class TemporalSelfAttention(nn.Module):
    """Self-Attention mechanism applied across the temporal dimension."""
    def __init__(self, dim, num_heads, mlp_dim, dim_head = 64,  dropout=0.):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads=num_heads, dim_head= dim_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, N, C)  -> (Batch, Time, Patches, Channels)
        Returns:
            Updated tensor after temporal attention.
        """
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)  # Reshape to (B * N, T, C)
        print(self.norm1(x).shape)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))  # FeedForward network
        
        x = x.reshape(B, N, T, C).permute(0, 2, 1, 3)  # Reshape back to (B, T, N, C)
        return x


class WindowAttention(nn.Module):
    """Standard Swin Transformer Window Attention."""
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.shape  # N should be window_size^2

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each (B_, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block that integrates spatial window attention."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size,
                                    qkv_bias=True, attn_drop=dropout,
                                    proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio * dim, dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, C)  -> (Batch, Patches, Channels)
            H, W: Height and Width of original feature map
        Returns:
            x: Updated tensor after Swin Transformer spatial processing.
        """
 
        B, L, C = x.shape
        #assert L == H * W, "Mismatch between input patches and spatial dimensions."
        H = W = int(L**0.5)
        # Reshape to 2D spatial form
        x = x.view(B, H, W, C)

        # Apply window shifting (if needed)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows * B, window_size, window_size, C)
        x_windows = x_windows.view(x_windows.shape[0], self.window_size * self.window_size, C)

        # Apply attention
        attn_windows = self.attn(x_windows)

        # Reverse partitioning
        attn_windows = attn_windows.view(x_windows.shape[0], self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Flatten back
        x = x.view(B, L, C)
        x = x + self.norm1(x)
        x = x + self.mlp(self.norm2(x))

        return x


class TemporalSwinTransformer(nn.Module):
    """Integrates temporal attention + Swin Transformer spatial block."""
    def __init__(self, dim, num_heads, window_size=7, dim_head = 64, shift_size=0, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.temporal_attn = TemporalSelfAttention(dim, num_heads, dropout, dim_head = dim_head)
        self.spatial_transformer = SwinTransformerBlock(dim, num_heads, window_size, shift_size, mlp_ratio, dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, N, C)  -> (Batch, Time, Patches, Channels)
        Returns:
            x: Updated tensor after temporal + spatial attention.
        """
        B, T, N, C = x.shape

        # Temporal Attention
        x = self.temporal_attn(x)

        # Flatten time dimension
        x = x.reshape(B * T, N, C)

        # Spatial Attention
        H = W = int(math.sqrt(N))  # Assuming square patches
        x = self.spatial_transformer(x, H, W)

        # Reshape back to (B, T, N, C)
        x = x.reshape(B, T, N, C)
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
        x = torch.cat((x,cls_temporal_tokens), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]

        H_out, W_out = self.num_patches_1d, self.num_patches_1d
        x = x.reshape(B,self.num_classes,  H_out, W_out, self.dim).permute(0, 1, 4, 2, 3)
        x = x.reshape(B * self.num_classes, H_out * W_out, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x)

        x = self.mlp_head(x.reshape(-1, self.dim))

        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)

        return x

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.encoder(x)
    

class SegmentationHeadSTViT(nn.Module):
        def __init__(self, input_sz, output_k, num_classes, num_patches_1d, patch_size):
            
            super(SegmentationHeadSTViT, self).__init__()
            self.heads = nn.Sequential(
                    nn.LayerNorm(input_sz),
                    nn.Linear(input_sz, output_k)
                )

            self.num_classes = num_classes
            self.num_patches_1d = num_patches_1d
            self.patch_size = patch_size

        def forward(self, x, b, h, w):
            results = []
            x = self.heads(x)
            x = x.reshape(b, self.num_patches_1d**2, self.patch_size**2, self.num_classes)
            x = x.reshape(b, h*w, self.num_classes)
            x = x.reshape(b, h, w, self.num_classes)

            return x
        
      
class DINOTSViT(nn.Module):
    """
    Modified TSViT that:
      1. Handles missing data.
      2. Uses continuous temporal positional encoding.
      3. Incorporates multi-scale fusion.
      4. Integrates a CNN encoder for low-level features.
    Additionally, the spatial transformer is decoupled from the number of classes.
    A separate segmentation head is used at the end.
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        
        self.dim = model_config['dim']
        self.time_window = model_config['time_window']
        self.temporal_depth = model_config.get('temporal_depth', model_config['depth']) # model_config['depth'] is fallback
        self.spatial_depth = model_config.get('spatial_depth', model_config['depth'])   # model_config['depth'] is fallback
        self.heads = model_config['heads']
        self.mlp_ratio = model_config.get('mlp_ratio', 4) # 
        mlp_dim = self.dim * self.mlp_ratio
        
        self.dim_head = model_config.get('dim_head', self.dim // self.heads)
        self.dropout_rate = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        
        self.pool = model_config['pool']
        
        self.num_tokens_spatial_cls = 1 if self.pool == 'cls' else 0 
        
        self.spatial_block = model_config['spatial_block']
        ##
        in_channels = (model_config['num_channels'] - 1)# Exclude time channel
        self.patch_dim_out  = in_channels*self.patch_size**2
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim_out, self.dim),
        )
        
        # token
        self.cls_token_spatial = nn.Parameter(torch.randn(1 , 1, self.dim))
        self.cls_temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        # 
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head, 
                                                mlp_dim, self.dropout_rate)
        
        # 
        num_patches = (self.image_size//self.patch_size) ** 2
        
        # temporal 
        self.temporal_embedding_layer = nn.Linear(self.time_window+2, self.dim) #+2 for 0 and potential out-of-range
        # spatial embeding
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches + self.num_tokens_spatial_cls, self.dim))
        
        if self.spatial_block == 'ViT':
            self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head,
                                            mlp_dim, self.dropout_rate)
        elif self.spatial_block == 'SwinViT':
            self.space_transformer = nn.Sequential(*[
                SwinTransformerBlock(
                    dim=self.dim,
                    num_heads=self.heads,
                    window_size=model_config.get('window_size', 4),
                    shift_size=0 if (i % 2 == 0) else model_config.get('window_size', 4) // 2,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout_rate
                ) for i in range(self.spatial_depth)
            ])
        
        
        self.dropout_emb = nn.Dropout(self.emb_dropout)
        
        self.norm_final = norm_layer(self.dim)
        
        
        self.head = nn.Identity()

        trunc_normal_(self.space_pos_embedding, std = 0.02)
        #trunc_normal_(self.cls_temporal_token, std = 0.02)
        nn.init.normal_(self.cls_token_spatial, std=1e-6)
        nn.init.normal_(self.cls_temporal_token, std=1e-6)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
                    
    def forward(self, x, return_patch_tokens = False):
        B, T, C, H, W = x.shape
        processed_temporal_outputs = []
        noprocessed= []
        for b_idx in range(B):
            single_sample_sits = x[b_idx].unsqueeze(dim = 0)
            single_sample_sits = self.temporal_transform(single_sample_sits)
            if single_sample_sits is not None: 
                #(1, NumPatches_H * NumPatches_W, Dim)
                processed_temporal_outputs.append(single_sample_sits)
            else:
                noprocessed.append(b_idx)

        if len(processed_temporal_outputs)==0: return None    
        x_spatial_input = torch.concat(processed_temporal_outputs, dim = 0)#self.temporal_transform(x)
        B_eff = x_spatial_input.shape[0]
        
        spatial_tokens_out = self.spatial_transform(x_spatial_input, H, W)
        
        if return_patch_tokens:
            if self.pool == 'cls':
                patch_tokens = self.norm_final(spatial_tokens_out)
                return patch_tokens[:, self.num_tokens_spatial_cls:, :], noprocessed 
            else: 
                return self.norm_final(spatial_tokens_out), noprocessed # Normalize all tokens

        else:
            if self.pool == 'cls':
                cls_output = self.norm_final(spatial_tokens_out)
                #spatial_tokens_out[:,0]
                return cls_output[:,0], noprocessed
            else:
                mean_output = spatial_tokens_out.mean(dim = 1)
                return self.norm_final(mean_output)
        
    def temporal_transform(self,single_sample):
        ## single sample to check in which positions there is no data
        B, T, C, H, W = single_sample.shape
        num_patches_spatial = (H // self.patch_size) * (W // self.patch_size)
        
        image_channels = single_sample[:, :, :C-1, :, :]  # (B, T, C-1, H, W)
        
        time_mask = (~(single_sample==0)).float()[:, :, 0, :, :].unsqueeze(dim=2)
        zero_channels = torch.all(time_mask == 0, dim=( 3,4))[0]
        notvalid_time_indices  = torch.where(zero_channels.squeeze())[0]
        ## filter those dates that does not have data
        retain_time_indices = [i for i in range(T) if i not in notvalid_time_indices ]
        if len(retain_time_indices)== 0: return None
        
        retained_image_channels  = image_channels[:,retain_time_indices, :, :, :]

        Teff = retained_image_channels.shape[1]
        
        x_patch_tokens = self.to_patch_embedding(retained_image_channels)
        # Reshape to (NumPatches_Total, T_eff, Dim)
        num_patches_spatial = (H // self.patch_size) * (W // self.patch_size)
        #x_patch_tokens = x_patch_tokens.reshape(num_patches_spatial, Teff, self.dim)
        
        # --- Temporal Positional Encoding ---
        time_indices_values = single_sample[:, retain_time_indices, -1, 0, 0] # (1, T_eff) values from time channel
        time_indices_values = time_indices_values.squeeze(0).to(torch.int64) # (T_eff)
        
        # Ensure indices are within bounds for one-hot encoding
        ##time_indices_values = torch.clamp(time_indices_values, 0, self.time_window + 1) # +1 for a potential max_value+1 category
        
        time_one_hot = F.one_hot(time_indices_values, num_classes=self.time_window + 2).to(x_patch_tokens.dtype) # (T_eff, time_window+2)
        temporal_pos_embed = self.temporal_embedding_layer(time_one_hot) # (T_eff, Dim)
        
        # Add temporal pos_embed to each patch's time series
        # x_patch_tokens: (NP_spatial, T_eff, Dim)
        # temporal_pos_embed: (T_eff, Dim) -> unsqueeze to (1, T_eff, Dim) for broadcasting
        x_patch_tokens = x_patch_tokens + temporal_pos_embed.unsqueeze(0)
        
        # --- Temporal Transformer ---
        # Prepend temporal CLS token to each patch's temporal sequence
        # temporal_token_cls: (1, 1, Dim)
        # Repeat for each spatial patch: (NP_spatial, 1, Dim)
        cls_temporal_tokens_repeated = repeat(self.cls_temporal_token, '() n d -> b n d', b=num_patches_spatial)
        
        # x_for_temporal_transformer: (NP_spatial, T_eff + 1, Dim)
        x_for_temporal_transformer = torch.cat((cls_temporal_tokens_repeated, x_patch_tokens), dim=1)
        x_for_temporal_transformer = self.dropout_emb(x_for_temporal_transformer) 
        # Output of temporal transformer: (NP_spatial, T_eff + 1, Dim)
        temporally_attended_features = self.temporal_transformer(x_for_temporal_transformer)
        
        # Aggregate over time: Use the output of the temporal CLS token
        # temporally_summarized_features shape: (NP_spatial, Dim)
        temporally_summarized_features = temporally_attended_features[:, 0] 

        # Reshape to (1, NP_spatial, Dim) for consistency before spatial transformer
        return temporally_summarized_features.unsqueeze(0)
    
    def interpolate_pos_encoding(self, x, w, h):
        
        previous_dtype = x.dtype
        
        
        if self.spatial_block == 'SwinViT':
            npatch = x.shape[1]
            N = self.space_pos_embedding.shape[1]
        else:
            npatch = x.shape[1] - 1
            N = self.space_pos_embedding.shape[1] - 1
            
        if npatch == N and w == h:
            return self.space_pos_embedding
        pos_embed = self.space_pos_embedding.float()
        if self.spatial_block == 'SwinViT':
            patch_pos_embed = pos_embed
        else:
                
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        #         # Simply specify an output size instead of a scale factor
        kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=False,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if self.spatial_block == 'SwinViT':
            return patch_pos_embed.to(previous_dtype)
        
        else:
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    
    def spatial_transform(self, x_input_patches, h,w):
        
        B_eff, NP_spatial, _ = x_input_patches.shape
        
        if self.pool == 'cls':
            # Prepend spatial CLS token: (B_eff, 1, D)
            cls_spatial_tokens_repeated = repeat(self.cls_token_spatial, '() n d -> b n d', b=B_eff)
            # x_for_spatial_transformer: (B_eff, NP_spatial + 1, D)
            x_for_spatial_transformer = torch.cat((cls_spatial_tokens_repeated, x_input_patches), dim=1)
        else: # 'mean' pool
            x_for_spatial_transformer = x_input_patches
        

        x_for_spatial_transformer = x_for_spatial_transformer + self.interpolate_pos_encoding(x_input_patches, h,w)
        x_for_spatial_transformer = self.dropout_emb(x_for_spatial_transformer)
                
        x_tokens = self.space_transformer(x_for_spatial_transformer)
        
        return x_tokens


class TSViT_SingleToken(nn.Module):
    """
    Modified TSViT that:
      1. Handles missing data.
      2. Uses continuous temporal positional encoding.
      3. Incorporates multi-scale fusion.
      4. Integrates a CNN encoder for low-level features.
    Additionally, the spatial transformer is decoupled from the number of classes.
    A separate segmentation head is used at the end.
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.time_window = model_config['time_window']
        self.temporal_depth = model_config.get('temporal_depth', model_config['depth'])
        self.spatial_depth = model_config.get('spatial_depth', model_config['depth'])
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout_rate = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        self.spatial_block = model_config['spatial_block']

        self.num_tokens = 1
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls or mean'
        
        # Hybrid CNN Encoder: process image channels (excluding the time channel).
        # Assuming model_config['num_channels'] includes the time channel, subtract one.
        #self.cnn_encoder = CNNEncoder(in_channels=model_config['num_channels'] - 1, out_channels=64)
        
        # New patch dimension: we now concatenate the CNN features with the original image channels and a mask.
        # New channels = (original image channels) + (cnn features) + (mask channel)
        in_channels = (model_config['num_channels'] - 1)# + 64  
        self.patch_dim = in_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
        )
        
        # Continuous temporal positional encoding layer.
        #self.temporal_embedding_layer = nn.Linear(1, self.dim)
        self.temporal_embedding_layer = nn.Linear(self.time_window+2, self.dim)
        
        # Instead of using a CLS token that depends on num_classes, we simply process patch tokens.
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # Temporal transformer that operates along the time dimension.
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout_rate)
        
        # After temporal transformer, aggregate over time (e.g., average pooling) to get per-patch features.

        num_patches = self.num_patches_1d ** 2
        
        
        
        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        # Spatial transformer (generic, not tied to num_classes).
        if self.pool == 'cls':
            self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches+ self.num_tokens, self.dim))
        elif self.pool == 'mean':
            self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        print(self.space_pos_embedding)
        if self.spatial_block == 'ViT':
            self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head,
                                             self.dim * self.scale_dim, self.dropout_rate)
        elif self.spatial_block == 'SwinViT':
            self.space_transformer = nn.Sequential(*[
                SwinTransformerBlock(
                    dim=self.dim,
                    num_heads=self.heads,
                    window_size=model_config.get('window_size', 4),
                    shift_size=0 if (i % 2 == 0) else model_config.get('window_size', 4) // 2,
                    mlp_ratio=model_config.get('mlp_ratio', 4),
                    dropout=self.dropout_rate
                ) for i in range(self.spatial_depth)
            ])
            
        self.dropout = nn.Dropout(self.emb_dropout)
        
        # Multi-scale fusion branch (using a 1D convolution over the patch tokens).
        self.scale_fusion = nn.Sequential(
            nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1),
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 4*self.dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*self.dim, self.dim * self.patch_size**2)
        )
        

        self.head = nn.Identity()
        self.norm = norm_layer(self.dim)

        trunc_normal_(self.space_pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        tmp_tr = []
        for b in range(B):
            n_img = x[b].unsqueeze(dim = 0)
            n_img = self.temporal_transform(n_img)
            if n_img is not None: tmp_tr.append(
                n_img)
    
        x = torch.concat(tmp_tr, dim = 0)#self.temporal_transform(x)
        new_num_patches = (H // self.patch_size) * (W // self.patch_size)
        x = x.view(len(tmp_tr), new_num_patches, self.dim)
        #x_fused = self.scale_fusion(x.transpose(1, 2)).transpose(1, 2)
        #x = x + x_fused
        
        if self.pool == 'cls':
            x = self.spatial_transform(x, H, W)
            x = self.norm(x)
            return x[:,0]
        else:
            x = self.spatial_transform(x, H, W)
            x = x.mean(dim =1)
            x = self.norm(x)
            return x
    
        
    def temporal_transform(self,x):
        
        B, T, C, H, W = x.shape
        image_channels = x[:, :, :C-1, :, :]  # (B, T, C-1, H, W)
        mask = (~(x==0)).float()
        mask_image = mask[:, :, 0, :, :]  
        mask_image = mask_image.unsqueeze(dim=2)
        zero_channels = torch.all(mask_image == 0, dim=( 3,4))[0]
        zero_channel_indices = torch.where(zero_channels.squeeze())[0]
        ## filter those dates that does not have data
        retain_channel_indices = [i for i in range(T) if i not in zero_channel_indices]
        if len(retain_channel_indices)== 0: return None
        image_channels = image_channels[:,retain_channel_indices]

        B, T, Cin, H, W = image_channels.shape
        
        #image_channels = image_channels.view(B * T, Cin, H, W)
        #cnn_features = self.cnn_encoder(image_channels)  # (B*T, 64, H, W)
        #cnn_features = cnn_features.view(B, T, 64, H, W)

        #plt.imshow(cnn_features[0,0,40].to('cpu').detach().numpy())
        #combined = torch.cat([image_channels.view(B, T, Cin, H, W), cnn_features], dim=2) 

        #x_patch = self.to_patch_embedding(combined)
        x_patch = self.to_patch_embedding(image_channels)
        
        ## temporal
        time_channel = x[:, retain_channel_indices, -1:, 0, 0]      # (B, T, 1, H, W)
        #time_channel = (time_channel * (self.time_window-1+.0001)).to(torch.int64)
        time_channel = time_channel.to(torch.int64)
        time_channel = F.one_hot(time_channel, num_classes=self.time_window+2).to(torch.float32)
        time_channel = time_channel.view(-1, self.time_window+2)
        temporal_pos_embedding = self.temporal_embedding_layer(time_channel).view(B, T, self.dim)
        # values
        x_patch = x_patch.reshape(B, -1, T, self.dim)
        x_patch += temporal_pos_embedding.unsqueeze(1)
        x_patch = x_patch.view(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * (H // self.patch_size) * (W // self.patch_size))
        x_patch = torch.cat((cls_temporal_tokens, x_patch), dim=1)

        x_patch = self.temporal_transformer(x_patch)
        x_patch = x_patch.mean(dim=1)
        
        return x_patch
    
    def interpolate_pos_encoding(self, x, w, h):
        
        previous_dtype = x.dtype
        
        
        if self.pool == 'mean':
            npatch = x.shape[1]
            N = self.space_pos_embedding.shape[1]
        else:
            npatch = x.shape[1] - 1
            N = self.space_pos_embedding.shape[1] - 1
            
        if npatch == N and w == h:
            return self.space_pos_embedding
        pos_embed = self.space_pos_embedding.float()
        if self.pool == 'mean':
            patch_pos_embed = pos_embed
        else:
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        #         # Simply specify an output size instead of a scale factor
        kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=False,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if self.pool == 'mean':
            return patch_pos_embed.to(previous_dtype)
        else:
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    
    def spatial_transform(self, x, h,w):
        
        if self.pool != 'mean':
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            #return self.space_transformer(x)
        
        x_tokens = x + self.interpolate_pos_encoding(x, h,w)
        x_tokens = self.dropout(x_tokens)
        x_tokens = self.space_transformer(x_tokens)
        
        return x_tokens