from ..base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class PoseTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PoseGaitNet(BaseModel):
    def build_network(self, model_cfg):
        in_channels = model_cfg["in_channels"]
        self.num_class = model_cfg['num_class']
        self.hidden_dim = 256
        self.out_dim = 768
        self.num_joints = model_cfg.get('V', 17)
        self.max_frames = model_cfg.get('T', 30)  # Changed to match your data
        
        # Initial embedding
        self.joint_embed = nn.Linear(2, self.hidden_dim)  # Changed to 2 for 2D coordinates
        
        # Positional encodings
        self.register_buffer('pos_encoding', 
            self._create_positional_encoding(self.num_joints, self.hidden_dim))
        self.register_buffer('temporal_encoding',
            self._create_positional_encoding(self.max_frames, self.hidden_dim))
        
        # Transformer blocks
        self.spatial_blocks = nn.ModuleList([
            PoseTransformerBlock(self.hidden_dim) for _ in range(4)
        ])
        
        self.temporal_blocks = nn.ModuleList([
            PoseTransformerBlock(self.hidden_dim) for _ in range(4)
        ])
        
        # Multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((2, 1)),
            nn.AdaptiveAvgPool2d((4, 1))
        ])
        
        # Final embedding
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 7, self.hidden_dim * 4),
            nn.LayerNorm(self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.out_dim),
            nn.LayerNorm(self.out_dim)
        )

    def _create_positional_encoding(self, length, dim):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        pose = ipts[0]
        N, C, T, V, M = pose.size()
        if len(pose.size()) == 5:
            pose = pose.squeeze(-1)
            
        # Process joints [N, 2, T, V] -> [N, T, V, 2]
        x = pose[:, :2, ...]  # Use only 2D coordinates
        x = x.permute(0, 2, 3, 1)  # [N, T, V, 2]
        
        # Embed joints
        x = self.joint_embed(x)  # [N, T, V, D]
        
        # Add positional encodings
        x = x + self.pos_encoding  # Add joint positions
        x = x + self.temporal_encoding.unsqueeze(2)  # Add temporal positions
        
        # Spatial transformer blocks
        N, T, V, D = x.shape
        x = x.reshape(N*T, V, D)
        for block in self.spatial_blocks:
            x = block(x)
        x = x.reshape(N, T, V, D)
        
        # Temporal transformer blocks
        x = x.transpose(1, 2).reshape(N*V, T, D)
        for block in self.temporal_blocks:
            x = block(x)
        x = x.reshape(N, V, T, D).transpose(1, 2)
        
        # Reshape for pooling [N, D, T, V]
        x = x.permute(0, 3, 1, 2)
        
        # Multi-scale feature extraction
        global_feat = self.global_pool(x).reshape(N, -1)
        local_feats = [pool(x).reshape(N, -1) for pool in self.local_pools]
        
        # Feature fusion
        x = torch.cat([global_feat] + local_feats, dim=1)
        embeddings = self.fusion(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        retval = {
            'training_feat': {
                'pose_contrastive_loss': {'embeddings': embeddings.unsqueeze(-1), 'labels': labs},
            },
            'visual_summary': {},
            'inference_feat': {
                'embeddings': embeddings.unsqueeze(-1)
            }
        }
        return retval