import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        channels = config['cnn']['channels']
        kernel_size = config['cnn']['kernel_size']
        use_batchnorm = config['cnn']['use_batchnorm']
        dropout = config['cnn']['dropout']
        num_classes = config['classifier']['num_classes']
        
        layers = []
        in_channels = 1
        
        for channel in channels:
            layers.append(nn.Conv1d(in_channels, channel, kernel_size, padding=kernel_size//2))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(channel))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout))
            in_channels = channel
        
        self.features = nn.Sequential(*layers)
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, config['classifier']['input_length'])
            out = self.features(dummy)
            flattened_dim = out.numel()
        
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        
        return logits, features