import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, num_classes=5, label_smoothing=0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)