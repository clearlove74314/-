import torch
import torch.nn as nn
from .feature_stream import FeatureStream
from .semantic_stream import SemanticStream
from .cross_attention import CrossAttention
from .diffusion_decoder import DiffusionDecoder


class DSDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.feature_stream = FeatureStream(
            conv_kernels=config['feature_stream']['conv_kernels'],
            channels=config['feature_stream']['channels'],
            use_channel_attention=config['feature_stream']['use_channel_attention'],
            dropout=config['feature_stream']['dropout']
        )
        
        self.semantic_stream = SemanticStream(
            num_classes=config['model']['num_classes'],
            embedding_dim=config['semantic_stream']['embedding_dim'],
            mlp_dims=config['semantic_stream']['mlp_dims'],
            label_smoothing=config['semantic_stream']['label_smoothing']
        )
        
        self.cross_attention = CrossAttention(
            num_heads=config['cross_attention']['num_heads'],
            dim=config['cross_attention']['dim'],
            dropout=config['cross_attention']['dropout'],
            use_layernorm=config['cross_attention']['use_layernorm'],
            use_residual=config['cross_attention']['use_residual']
        )
        
        self.diffusion_decoder = DiffusionDecoder(
            input_length=config['model']['input_length'],
            timesteps=config['diffusion']['timesteps'],
            noise_schedule=config['diffusion']['noise_schedule'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end']
        )
        
        self.fusion_proj = nn.Linear(config['model']['feature_dim'], config['model']['input_length'])
    
    def forward(self, x, labels, t=None):
        batch_size = x.shape[0]
        
        feature_out, _ = self.feature_stream(x)
        feature_out = feature_out.mean(dim=-1)
        
        semantic_out, _ = self.semantic_stream(labels)
        semantic_out = semantic_out.unsqueeze(1)
        
        attn_out, attn_weights = self.cross_attention(
            query=feature_out.unsqueeze(1),
            key=semantic_out,
            value=semantic_out
        )
        attn_out = attn_out.squeeze(1)
        
        condition = attn_out
        
        if t is None:
            t = torch.randint(0, self.diffusion_decoder.timesteps, (batch_size,)).to(x.device)
        
        noise = torch.randn_like(x)
        x_noisy, _ = self.diffusion_decoder.add_noise(x, t, noise)
        
        predicted_noise = self.diffusion_decoder(x_noisy.unsqueeze(1), t, condition)
        
        return predicted_noise.squeeze(1), noise, attn_weights
    
    def generate(self, labels, num_samples=1, device='cuda'):
        semantic_out, _ = self.semantic_stream(labels)
        semantic_out = semantic_out.unsqueeze(1)
        
        dummy_feature = torch.randn(num_samples, self.config['model']['feature_dim']).to(device)
        attn_out, _ = self.cross_attention(
            query=dummy_feature.unsqueeze(1),
            key=semantic_out,
            value=semantic_out
        )
        condition = attn_out.squeeze(1)
        
        generated = self.diffusion_decoder.sample(condition, num_samples, device)
        return generated