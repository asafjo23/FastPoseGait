import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

from ..base_model import BaseModel


class SpatialAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SpatialAttentionModule, self).__init__()
        self.attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # Self-attention over the spatial dimension (joints)
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class PoseTemporalModel(BaseModel):

    def build_network(self, model_cfg):
        in_channels = model_cfg["in_channels"]
        num_classes= model_cfg['num_class']
        hrnet_output_dim = 1280
        spatial_dim = 256
        temporal_hidden_dim = 512
        num_heads = 4

        self.hrnet = self.get_hrnet_backbone()
        self.fc_spatial = nn.Linear(hrnet_output_dim, spatial_dim)
        self.spatial_attention = SpatialAttentionModule(embed_dim=spatial_dim, num_heads=num_heads)
        self.lstm = nn.LSTM(input_size=spatial_dim, hidden_size=temporal_hidden_dim, num_layers=2, batch_first=True)
        self.fc_temporal = nn.Linear(temporal_hidden_dim, 128)
        self.fc_output = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def get_hrnet_backbone(self, input_channels=2):
        # Load HRNet and replace the first conv layer to accept input_channels
        import timm
        hrnet = timm.create_model('hrnet_w18', pretrained=True)

        # Replace first convolutional layer to accept 2D joint locations (2 channels)
        hrnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        hrnet.fc = nn.Identity()  # Remove the classification head
        return hrnet

    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        pose = ipts[0]
        if len(pose.size()) == 5:  # [N, C, T, V, M]
            pose = pose.squeeze(-1)

        # Input x has shape (B, C, T, V, M) - squeeze the M dimension
        x = pose  # Shape: (B, C, T, V)

        # Permute to (B, T, V, C) for HRNet processing
        x = x.permute(0, 2, 3, 1)  # Shape: (batch_size, seq_len, num_joints, channels)

        batch_size, seq_len, num_joints, hrnet_input_dim = x.size()

        # Reshape to process each joint feature separately with HRNet
        x = x.reshape(batch_size * seq_len * num_joints, hrnet_input_dim, 1, 1)  # Adding height, width as (1, 1) for HRNet

        # Pass through HRNet
        x = self.hrnet(x)

        # Flatten and reshape
        x = x.reshape(batch_size, seq_len, num_joints, -1)  # Shape: (B, T, V, hrnet_output_dim)

        # Apply fully connected layer to reduce HRNet output dimension
        x = self.fc_spatial(x)

        # Reshape to treat each joint as a spatial "token" for attention
        x = x.view(batch_size * seq_len, num_joints, -1)

        # Spatial attention module
        x = self.spatial_attention(x)  # Shape: (B * T, V, spatial_dim)

        # Reduce dimensions along the joints (spatial pooling)
        x = x.mean(dim=1)  # Shape: (B * T, spatial_dim)

        # Reshape back for temporal processing
        x = x.view(batch_size, seq_len, -1)

        # Temporal LSTM layer
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch_size, temporal_hidden_dim)

        # Get the last layer's hidden state
        temporal_features = h_n[-1]  # Shape: (B, temporal_hidden_dim)

        # Fully connected layer for embeddings (used in metric learning)
        embeddings = self.fc_embedding(temporal_features)
        embeddings = self.relu(embeddings)
        embeddings = self.dropout(embeddings)

        # Classification layer for logits
        logits = self.fc_output(embeddings)

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embeddings.unsqueeze(-1), 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
            },
            'visual_summary': {},
            'inference_feat': {
                'embeddings': embeddings.unsqueeze(-1)
            }
        }
        return retval