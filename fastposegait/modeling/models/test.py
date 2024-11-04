from torch.nn.functional import embedding

from ..base_model import BaseModel

import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from ..graph import Graph


class Test(BaseModel):

    def build_network(self, model_cfg):
        in_channels = model_cfg["in_channels"]
        self.num_class = model_cfg['num_class']

        # Model dimensions
        self.hidden_dim = 64
        self.attn_dim = 128
        self.out_dim = 256

        # Spatial branch - processes joints relationships
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels[0], self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )

        # Temporal branch - processes motion over time
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels[0], self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )

        # Cross attention layers
        self.spatial_to_temporal = CrossAttention(self.hidden_dim, self.attn_dim)
        self.temporal_to_spatial = CrossAttention(self.hidden_dim, self.attn_dim)

        # Final embedding layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(self.out_dim, self.num_class)

    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        pose = ipts[0]
        if len(pose.size()) == 4:
            pose = pose.unsqueeze(2)

        N, C, T, V, M = pose.size()
        pose = pose.squeeze(-1)  # Remove M dimension if M=1

        # Process spatial dimension (N, C, T, V)
        spatial_x = pose.permute(0, 1, 2, 3).contiguous()
        spatial_x = self.spatial_conv(spatial_x)

        # Process temporal dimension (N, C, V, T)
        temporal_x = pose.permute(0, 1, 3, 2).contiguous()
        temporal_x = self.temporal_conv(temporal_x)

        # Cross attention between spatial and temporal features
        spatial_attended = self.temporal_to_spatial(spatial_x, temporal_x)
        temporal_attended = self.spatial_to_temporal(temporal_x, spatial_x)

        # Combine features
        spatial_feat = spatial_attended.mean(dim=(2, 3))  # Global pooling
        temporal_feat = temporal_attended.mean(dim=(2, 3))  # Global pooling

        # Concatenate and generate embedding
        combined_feat = torch.cat([spatial_feat, temporal_feat], dim=1)
        embeddings = self.fc(combined_feat)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Generate logits
        logits = self.classifier(embeddings)

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embeddings.unsqueeze(-1), 'labels': labs},
                'softmax': {'logits': logits.unsqueeze(-1), 'labels': labs}
            },
            'visual_summary': {
                'image/pose': pose.view(N * T, M, V, C)
            },
            'inference_feat': {
                'embeddings': embeddings.unsqueeze(-1)
            }
        }
        return retval


class CrossAttention(nn.Module):
    def __init__(self, in_dim, attn_dim):
        super().__init__()
        self.query_proj = nn.Conv2d(in_dim, attn_dim, 1)
        self.key_proj = nn.Conv2d(in_dim, attn_dim, 1)
        self.value_proj = nn.Conv2d(in_dim, attn_dim, 1)
        self.scale = attn_dim ** -0.5

        self.out_proj = nn.Sequential(
            nn.Conv2d(attn_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )

    def forward(self, x, context):
        # x: (N, C, H, W)
        # context: (N, C, H', W')
        N = x.size(0)

        # Project queries, keys, and values
        q = self.query_proj(x).view(N, -1, x.size(2) * x.size(3))  # (N, C, H*W)
        k = self.key_proj(context).view(N, -1, context.size(2) * context.size(3))  # (N, C, H'*W')
        v = self.value_proj(context).view(N, -1, context.size(2) * context.size(3))  # (N, C, H'*W')

        # Compute attention
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale  # (N, H*W, H'*W')
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (N, C, H*W)
        out = out.view(N, -1, x.size(2), x.size(3))  # (N, C, H, W)

        return self.out_proj(out) + x  # Add residual connection
