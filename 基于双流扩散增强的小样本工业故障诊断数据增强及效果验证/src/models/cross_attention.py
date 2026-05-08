import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, num_heads=4, dim=256, dropout=0.1, use_layernorm=True, use_residual=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
        if use_layernorm:
            self.norm = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        k = self.key_proj(key).view(batch_size, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        v = self.value_proj(value).view(batch_size, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim // self.num_heads) ** 0.5
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.output_proj(output)
        
        if self.use_residual:
            output = output + query
        
        if self.use_layernorm:
            output = self.norm(output)
        
        return output, attn