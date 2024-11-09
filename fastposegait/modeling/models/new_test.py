import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


from ..base_model import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant 'pe' matrix with values dependent on
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i+1
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class GaitRecognitionModel(BaseModel):

    def build_network(self, model_cfg):
        d_model = 128
        nhead = 8
        num_layers = 4
        dropout = 0.1

        self.joint_embedding = nn.Linear(10, d_model)  # 2D coordinates to d_model

        # Positional Encoding
        self.spatial_pos_encoder = PositionalEncoding(d_model)
        self.temporal_pos_encoder = PositionalEncoding(d_model)

        # Spatial Transformer Encoder
        spatial_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.spatial_transformer = nn.TransformerEncoder(spatial_encoder_layer, num_layers=num_layers)

        # Temporal Transformer Encoder
        temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=num_layers)

        # Classification Head
        self.fc = nn.Linear(d_model, 100)

    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        pose = ipts[0]
        if len(pose.size()) == 5:  # [N, C, T, V, M]
            pose = pose.squeeze(-1)
        # x shape: (B, C, T, V)
        x = pose
        B, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1)  # (B, T, V, C)

        # Flatten coordinates and embed joints
        x = x.reshape(B * T, V, C)  # (B*T, V, C)
        joint_embeddings = self.joint_embedding(x)  # (B*T, V, d_model)
        joint_embeddings = self.spatial_pos_encoder(joint_embeddings)  # (B*T, V, d_model)

        # Prepare for Transformer (requires shape: (sequence_length, batch_size, d_model))
        joint_embeddings = joint_embeddings.permute(1, 0, 2)  # (V, B*T, d_model)

        # Spatial Transformer
        spatial_output = self.spatial_transformer(joint_embeddings)  # (V, B*T, d_model)
        spatial_output = spatial_output.permute(1, 0, 2)  # (B*T, V, d_model)

        # Pool over joints to get frame-level features
        frame_features = spatial_output.mean(dim=1)  # (B*T, d_model)
        frame_features = frame_features.reshape(B, T, -1)  # (B, T, d_model)

        # Temporal Positional Encoding
        frame_features = self.temporal_pos_encoder(frame_features)  # (B, T, d_model)

        # Prepare for Temporal Transformer
        frame_features = frame_features.permute(1, 0, 2)  # (T, B, d_model)

        # Temporal Transformer
        temporal_output = self.temporal_transformer(frame_features)  # (T, B, d_model)
        temporal_output = temporal_output.permute(1, 0, 2)  # (B, T, d_model)

        # Pool over time to get sequence-level features
        embeddings = temporal_output.mean(dim=1)  # (B, d_model)

        # Classification
        logits = self.fc(embeddings)  # (B, num_classes)
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embeddings.unsqueeze(-1), 'labels': labs},
                'softmax': {'logits': logits.unsqueeze(-1), 'labels': labs},
            },
            'visual_summary': {},
            'inference_feat': {
                'embeddings': embeddings.unsqueeze(-1)
            }
        }
        return retval