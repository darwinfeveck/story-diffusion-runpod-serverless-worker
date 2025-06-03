import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x

class FuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(self, prompt_embeds, id_embeds, class_tokens_mask=None):
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        return self.fuse_fn(prompt_embeds, id_embeds)

class PerceiverResampler(nn.Module):
    def __init__(self, input_dim, num_layers=4, num_latents=64, latent_dim=512):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.MultiheadAttention(latent_dim, num_heads=8, batch_first=True),  # Fixed
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim * 4),
                nn.GELU(),
                nn.Linear(latent_dim * 4, latent_dim),
            ) for _ in range(num_layers)
        ])
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.norm_out = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = self.input_proj(x)
        latents = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        for layer in self.layers:
            latents = layer(latents + x)
        return self.norm_out(latents)

class QFormer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True),  # Fixed
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim * 4),
                nn.GELU(),
                nn.Linear(input_dim * 4, input_dim),
            ) for _ in range(6)
        ])
        self.output_proj = nn.Linear(input_dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm_out(self.output_proj(x))

class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.perceiver_resampler = PerceiverResampler(input_dim=1024)
        self.qformer = QFormer(input_dim=512, output_dim=768)
        self.fuse_module = FuseModule(2048)

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask=None):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)
        shared_id_embeds = self.vision_model(id_pixel_values)[1]
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)
        id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)
        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        
        # V2 processing
        id_embeds = id_embeds.view(b, num_inputs, -1)
        id_embeds = self.perceiver_resampler(id_embeds)
        id_embeds = self.qformer(id_embeds)
        id_embeds = id_embeds.unsqueeze(2)
        
        return self.fuse_module(prompt_embeds, id_embeds)