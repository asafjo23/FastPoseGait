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


class GaitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Replace spatial conv with FC-based module
        self.spatial_module = JointsModule(in_channels, out_channels)
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.spatial_attention = SpatialAttention(out_channels)
        self.temporal_attention = TemporalAttention(out_channels)

    def forward(self, x):
        # Spatial relations
        x = self.spatial_module(x)
        x = self.spatial_attention(x)

        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.temporal_attention(x)

        x = self.bn(x)
        x = self.relu(x)
        return x


class Test(BaseModel):
    def build_network(self, model_cfg):
        in_channels = model_cfg["in_channels"]
        self.num_class = model_cfg['num_class']
        self.hidden_dim = 64
        self.out_dim = 256

        # Input normalization
        self.input_bn = nn.BatchNorm2d(in_channels[0])

        # Spatial-Temporal Graph Convolution blocks
        self.st_blocks = nn.Sequential(
            GaitBlock(in_channels[0], self.hidden_dim),
            nn.MaxPool2d(kernel_size=(2, 1)),
            GaitBlock(self.hidden_dim, self.hidden_dim * 2),
            nn.MaxPool2d(kernel_size=(2, 1)),
            GaitBlock(self.hidden_dim * 2, self.hidden_dim * 4)
        )

        # Multi-scale feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((2, 1)),
            nn.AdaptiveAvgPool2d((4, 1))
        ])

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 4 * 7, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.out_dim),
            nn.BatchNorm1d(self.out_dim)
        )

    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        pose = ipts[0]
        N, C, T, V, M = pose.size()
        if len(pose.size()) == 5:  # [N, C, T, V, M]
            pose = pose.squeeze(-1)

        del ipts
        x = pose[:, :2, ...]
        # Input normalization
        x = self.input_bn(x)

        # Process with ST-GCN blocks
        x = self.st_blocks(x)

        # Multi-scale feature extraction
        global_feat = self.global_pool(x).contiguous().reshape(N, -1)
        local_feats = [pool(x).contiguous().reshape(N, -1) for pool in self.local_pools]

        # Feature fusion
        x = torch.cat([global_feat] + local_feats, dim=1)
        embeddings = self.fusion(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embeddings.unsqueeze(-1), 'labels': labs},
            },
            'visual_summary': {
            },
            'inference_feat': {
                'embeddings': embeddings.unsqueeze(-1)
            }
        }
        return retval
