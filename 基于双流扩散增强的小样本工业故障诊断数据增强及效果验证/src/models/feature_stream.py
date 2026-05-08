import torch
import torch.nn as nn
from einops import rearrange


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
    
    def forward(self, x):
        batch, channels, length = x.shape
        y = self.avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1)
        return x * torch.sigmoid(y)


class FeatureStream(nn.Module):
    def __init__(self, conv_kernels=[3, 5, 9], channels=[64, 128, 256], 
                 use_channel_attention=True, dropout=0.1):
        super().__init__()
        self.conv_kernels = conv_kernels
        self.channels = channels
        self.use_channel_attention = use_channel_attention
        
        self.layers = nn.ModuleList()
        
        for i, (kernel, channel) in enumerate(zip(conv_kernels, channels)):
            if i == 0:
                in_channels = 1
            else:
                in_channels = channels[i-1]
            
            layer = nn.Sequential(
                nn.Conv1d(in_channels, channel, kernel, padding=kernel//2),
                nn.BatchNorm1d(channel),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)
            
            if use_channel_attention:
                self.layers.append(ChannelAttention(channel))
        
        self.final_conv = nn.Conv1d(channels[-1], 256, 1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = []
        
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        
        x = self.final_conv(x)
        
        return x, features