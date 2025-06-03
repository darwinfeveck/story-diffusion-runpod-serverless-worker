import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from typing import Optional, Tuple

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}

class MLP(nn.Module):
    """V2 MLP with LayerNorm and residual connections"""
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, activation=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))

class V2Attention(nn.Module):
    """V2-style attention block"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
    def forward(self, x):
        x = self.norm(x)
        return self.attn(x, x, x)[0]

class V2Block(nn.Module):
    """Complete V2 transformer block"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.attn = V2Attention(dim, num_heads)
        self.mlp = MLP(dim, hidden_dim=dim*mlp_ratio)
        
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class QFormerPerceiver(nn.Module):
    """V2 Q-Former with perceiver architecture"""
    def __init__(self, input_dim=1024, latent_dim=512, num_latents=64, depth=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.proj_in = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.ModuleList([
            V2Block(latent_dim) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        x = self.proj_in(x)
        latents = self.latents.expand(x.size(0), -1, -1)
        
        for block in self.blocks:
            latents = block(latents + x)
            
        return self.norm_out(latents)

class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    """Complete V2 ID Encoder"""
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.qformer_perceiver = QFormerPerceiver()
        self.final_proj = nn.Linear(512, 2048)
        
    def forward(self, id_pixel_values, prompt_embeds):
        # Process ID images
        b, num_inputs = id_pixel_values.shape[:2]
        id_pixel_values = id_pixel_values.view(b*num_inputs, *id_pixel_values.shape[2:])
        
        # Get CLIP embeddings
        clip_out = self.vision_model(id_pixel_values)
        pooled = clip_out[1]
        
        # Dual projections
        embeds1 = self.visual_projection(pooled)
        embeds2 = self.visual_projection_2(pooled)
        
        # Reshape and process
        x = torch.cat([embeds1, embeds2], dim=-1)
        x = x.view(b, num_inputs, -1)
        x = self.qformer_perceiver(x)
        
        # Final projection
        return self.final_proj(x)