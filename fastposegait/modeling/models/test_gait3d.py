from torch.nn.functional import embedding

from ..base_model import BaseModel

import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from ..graph import Graph


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, C, T, V]
        b, c, t, v = x.size()
        y = self.avg_pool(x).view(b, c, t)
        y = self.fc(y.transpose(1, 2)).transpose(1, 2)
        y = y.view(b, c, t, 1)
        return x * y


class JointsModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # FC layer to process all joints together
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [B, C, T, V]
        B, C, T, V = x.size()
        
        # Reshape to process joints with FC: [B*T, V, C]
        x = x.permute(0, 2, 3, 1).reshape(-1, V, C)
        
        # Apply FC: [B*T, V, C_out]
        x = self.fc(x)
        
        # Reshape back: [B, C_out, T, V]
        x = x.reshape(B, T, V, self.out_channels).permute(0, 3, 1, 2)
        
        x = self.bn(x)
        x = self.relu(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, T, V = x.size()
        # Reshape to [B*T, V, C]
        x = x.permute(0, 2, 3, 1).reshape(-1, V, C)
        
        # Multi-head attention
        qkv = self.qkv(x).reshape(-1, V, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, V, C)
        x = self.proj(x)
        
        # Reshape back to [B, C, T, V]
        x = x.reshape(B, T, V, C).permute(0, 3, 1, 2)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(channels, channels, (3, 1), dilation=(2, 1), padding=(2, 0))
        self.attention = TemporalAttention(channels)
        
    def forward(self, x):
        local = self.conv1(x)
        global_feat = self.conv2(x)
        out = local + global_feat
        return self.attention(out)


class EnhancedGaitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.joint_attention = MultiHeadSelfAttention(in_channels)
        self.temporal_encoder = TemporalEncoder(out_channels)
        
        # Two-stream processing
        self.spatial_stream = nn.Sequential(
            JointsModule(in_channels, out_channels),
            SpatialAttention(out_channels)
        )
        
        self.temporal_stream = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            self.temporal_encoder
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Residual
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.residual(x)
        
        # Joint relationships
        joints = self.joint_attention(x)
        
        # Two-stream processing
        spatial = self.spatial_stream(joints)
        temporal = self.temporal_stream(x)
        
        # Feature fusion
        out = self.fusion(torch.cat([spatial, temporal], dim=1))
        
        # Residual connection
        out = self.bn(out + identity)
        out = self.relu(out)
        return out


class TestGait3D(BaseModel):
    def build_network(self, model_cfg):
        in_channels = model_cfg["in_channels"]
        self.hidden_dim = 64
        self.out_dim = 256

        # Input normalization
        self.input_bn = nn.BatchNorm2d(in_channels[0])

        # Spatial attention for joints
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels[0], self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, 1, 1),
            nn.Sigmoid()
        )

        # Main feature extraction
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels[0], self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, (3,1), padding=(1,0)),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            
            # Block 2
            nn.Conv2d(self.hidden_dim, self.hidden_dim*2, 1),
            nn.BatchNorm2d(self.hidden_dim*2),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim*2, self.hidden_dim*2, (3,1), padding=(1,0)),
            nn.BatchNorm2d(self.hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            
            # Block 3
            nn.Conv2d(self.hidden_dim*2, self.hidden_dim*4, 1),
            nn.BatchNorm2d(self.hidden_dim*4),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim*4, self.hidden_dim*4, (3,1), padding=(1,0)),
            nn.BatchNorm2d(self.hidden_dim*4),
            nn.ReLU()
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        pose = ipts[0]
        
        # Remove M dimension if present
        if len(pose.size()) == 5:
            pose = pose.squeeze(-1)
        
        N, C, T, V = pose.size()
        
        # Input normalization
        x = self.input_bn(pose)
        
        # Apply spatial attention
        att = self.spatial_attention(x)
        x = x * att
        
        # Feature extraction
        x = self.backbone(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(N, -1)
        
        # Feature fusion
        embeddings = self.fusion(x)
        
        return {
            'training_feat': {
                'triplet': {'embeddings': embeddings.unsqueeze(-1), 'labels': labs},
            },
            'visual_summary': {
            },
            'inference_feat': {
                'embeddings': embeddings.unsqueeze(-1)
            }
        }
