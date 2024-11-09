from ..base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbed(nn.Module):
    """Split pose sequence into patches and embed them"""
    def __init__(self, time_patch_size=2, joint_patch_size=1, in_channels=2, embed_dim=256):
        super().__init__()
        self.time_patch_size = time_patch_size
        self.joint_patch_size = joint_patch_size
        
        # Calculate patch dimension
        self.patch_dim = time_patch_size * joint_patch_size * in_channels
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, T, V]
        B, C, T, V = x.shape
        
        # Ensure dimensions are divisible by patch sizes
        assert T % self.time_patch_size == 0, f"Sequence length {T} must be divisible by patch size {self.time_patch_size}"
        assert V % self.joint_patch_size == 0, f"Number of joints {V} must be divisible by patch size {self.joint_patch_size}"
        
        # Reshape into patches
        x = x.permute(0, 2, 3, 1)  # [B, T, V, C]
        x = x.reshape(B, T // self.time_patch_size, self.time_patch_size, 
                     V // self.joint_patch_size, self.joint_patch_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, -1, self.patch_dim)
        
        # Project patches
        x = self.proj(x)
        x = self.norm(x)
        return x

class PoseViT(BaseModel):
    def build_network(self, model_cfg):
        in_channels = model_cfg["in_channels"][0]  # Usually 2 for 2D coordinates
        self.num_class = model_cfg['num_class']
        self.hidden_dim = 256
        self.out_dim = 1024
        num_heads = 8
        num_layers = 12
        mlp_ratio = 4
        
        # Calculate number of patches
        T = model_cfg.get('T', 30)
        V = model_cfg.get('V', 17)
        self.time_patch_size = 2
        self.joint_patch_size = 1
        
        # Calculate exact number of patches
        self.num_patches = (T // self.time_patch_size) * (V // self.joint_patch_size)
        print(f"Number of patches: {self.num_patches}")  # Debug print
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            time_patch_size=self.time_patch_size,
            joint_patch_size=self.joint_patch_size,
            in_channels=in_channels,
            embed_dim=self.hidden_dim
        )
        
        # Add [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        
        # Position embedding - make sure dimensions match
        num_positions = self.num_patches + 1  # Add 1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, 766, self.hidden_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * mlp_ratio,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.out_dim),
        )
        
        # Initialize weights
        self._init_weights()
        
        # Print model configuration
        print(f"Model configuration:")
        print(f"Input shape: C={in_channels}, T={T}, V={V}")
        print(f"Patch size: Time={self.time_patch_size}, Joints={self.joint_patch_size}")
        print(f"Number of patches: {self.num_patches}")
        print(f"Position embedding shape: {self.pos_embed.shape}")
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embed
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.proj.bias)

    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        pose = ipts[0]
        N, C, T, V, M = pose.size()
        if len(pose.size()) == 5:
            pose = pose.squeeze(-1)  # Remove M dimension
            
        # Patch embedding [B, num_patches, hidden_dim]
        x = self.patch_embed(pose)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding and dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Get CLS token representation
        x = x[:, 0]
        
        # MLP head
        embeddings = self.mlp_head(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        retval = {
            'training_feat': {
                'pose_vit': {'embeddings': embeddings.unsqueeze(-1), 'labels': labs},
            },
            'visual_summary': {},
            'inference_feat': {
                'embeddings': embeddings.unsqueeze(-1)
            }
        }
        return retval