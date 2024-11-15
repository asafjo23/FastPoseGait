from ..base_model import BaseModel

import torch.nn as nn
import torch.nn.functional as F
import torch


from ..components import SeparateBNNecks, SeparateFCs


class JointsModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1).reshape(-1, V, C)
        x = self.fc(x)
        x = x.reshape(B, T, V, self.out_channels).permute(0, 3, 1, 2)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # First reshape and mix all joints' features
        self.joint_mixer = nn.Conv2d(channels, channels, kernel_size=(1, 17))
        # Temporal convolution considering mixed joint features
        self.temporal_conv = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0)
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [B, C, T, V] where V is number of joints (17)
        # Mix information from all joints
        x_mixed = self.joint_mixer(x)  # Result: [B, C, T, 1]
        # Expand back to all joints
        x_mixed = x_mixed.expand(-1, -1, -1, 17)  # [B, C, T, V]
        # Add the mixed information to original features
        x = x + x_mixed
        # Apply temporal convolution
        x = self.temporal_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ViewAwareAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.view_embed = nn.Parameter(torch.randn(1, 1, channels))
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        B, C, T, V = x.size()
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, T*V, C)
        x_with_view = x_reshaped + self.view_embed
        attn_out, _ = self.mha(
            query=x_with_view,
            key=x_with_view,
            value=x_with_view
        )
        x1 = self.norm1(x_reshaped + attn_out)
        ffn_out = self.ffn(x1)
        out = self.norm2(x1 + ffn_out)
        out = out.reshape(B, T, V, C).permute(0, 3, 1, 2)
        
        return out


class GaitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.spatial_module = JointsModule(in_channels, out_channels)
        self.temporal_module = TemporalFeatureExtractor(out_channels)
        self.view_attention = ViewAwareAttention(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.spatial_module(x)
        x = self.dropout(x)
        x = self.temporal_module(x)
        x = self.view_attention(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Test(BaseModel):
    def build_network(self, model_cfg):
        dropout_rate = 0.1
        in_channels = model_cfg["in_channels"]
        self.num_class = model_cfg["num_class"]
        self.hidden_dim = 64
        self.out_dim = 256
        self.input_bn = nn.BatchNorm2d(in_channels[0])
        self.st_blocks = nn.Sequential(
            GaitBlock(in_channels[0], self.hidden_dim),
            nn.Dropout2d(p=dropout_rate),
            nn.MaxPool2d(kernel_size=(2, 1), padding=(1, 0)),
            GaitBlock(self.hidden_dim, self.hidden_dim * 2),
            nn.Dropout2d(p=dropout_rate),
            nn.MaxPool2d(kernel_size=(2, 1), padding=(1, 0)),
            GaitBlock(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.Dropout2d(p=dropout_rate),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_pools = nn.ModuleList(
            [nn.AdaptiveAvgPool2d((2, 1)), nn.AdaptiveAvgPool2d((4, 1))]
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 4 * 7, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.hidden_dim * 4, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
        )

        self.head = SeparateFCs(parts_num=17, in_channels=256, out_channels=256)
        self.BNNecks = SeparateBNNecks(
            class_num=self.num_class, in_channels=self.out_dim, parts_num=17
        )

    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        pose = ipts[0]
        N, C, T, V, M = pose.size()
        if len(pose.size()) == 5:  # [N, C, T, V, M]
            pose = pose.squeeze(-1)

        del ipts
        x = pose
        x = self.input_bn(x)
        x = self.st_blocks(x)

        global_feat = self.global_pool(x).contiguous().reshape(N, -1)
        local_feats = [pool(x).contiguous().reshape(N, -1) for pool in self.local_pools]
        x = torch.cat([global_feat] + local_feats, dim=1)
        embeddings = self.fusion(x)
        embeddings = self.head(embeddings.unsqueeze(-1))
        embed_2, logits = self.BNNecks(embeddings)

        retval = {
            "training_feat": {
                "tuplet": {"embeddings": embeddings, "labels": labs},
                "snr": {"embeddings": embeddings, "labels": labs},
                "softmax": {"logits": logits, "labels": labs}
            },
            "visual_summary": {},
            "inference_feat": {"embeddings": embed_2},
        }
        return retval
