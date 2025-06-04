import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig
from typing import Optional

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=None, activation=nn.GELU):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim * 4
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))

class V2AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class PerceiverResamplerV2(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=512, num_latents=64, depth=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.proj_in = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.ModuleList([
            V2AttentionBlock(latent_dim) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = self.proj_in(x)
        latents = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        for block in self.blocks:
            latents = block(latents + x)
        return self.norm_out(latents)

class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        config = CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_hidden_layers=24,
            patch_size=14,
            projection_dim=768
        )
        super().__init__(config)
        
        # V2-specific components
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.qformer_perceiver = nn.ModuleDict({
            'perceiver_resampler': PerceiverResamplerV2(),
            'token_proj': nn.Sequential(
                nn.Linear(2048, 2048),  # 1024 (CLIP) + 1280 (projection_2)
                nn.LayerNorm(2048)
            )
        })
        self.final_proj = nn.Linear(512, 2048)  # Matches V2 checkpoint

    def forward(self, id_pixel_values, prompt_embeds):
        # Process ID images
        batch_size, num_inputs = id_pixel_values.shape[:2]
        id_pixel_values = id_pixel_values.view(batch_size * num_inputs, *id_pixel_values.shape[2:])
        
        # Get CLIP features
        clip_out = self.vision_model(id_pixel_values)
        pooled = clip_out[1]  # [batch*num_inputs, 1024]
        
        # Dual projections
        embeds1 = self.visual_projection(pooled)  # Original CLIP projection
        embeds2 = self.visual_projection_2(pooled)  # V2 additional projection
        
        # Combine and process through V2 blocks
        combined = torch.cat([embeds1, embeds2], dim=-1)  # [batch*num_inputs, 2048]
        x = self.qformer_perceiver['token_proj'](combined)
        x = x.view(batch_size, num_inputs, -1)
        x = self.qformer_perceiver['perceiver_resampler'](x)
        
        # Final projection
        return self.final_proj(x)  # [batch_size, num_latents, 2048]