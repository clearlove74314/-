import torch
import torch.nn as nn


class DiffusionLoss(nn.Module):
    def __init__(self, loss_type='l2'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predicted_noise, target_noise):
        return self.loss_fn(predicted_noise, target_noise)