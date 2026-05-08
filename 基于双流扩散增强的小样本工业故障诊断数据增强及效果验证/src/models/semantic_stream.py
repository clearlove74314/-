import torch
import torch.nn as nn


class SemanticStream(nn.Module):
    def __init__(self, num_classes=5, embedding_dim=128, mlp_dims=[128, 256, 256], 
                 label_smoothing=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        layers = []
        in_dim = embedding_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        
        self.mlp = nn.Sequential(*layers)
        self.label_smoothing = label_smoothing
    
    def forward(self, labels):
        embeddings = self.embedding(labels)
        semantic_features = self.mlp(embeddings)
        return semantic_features, embeddings
    
    def get_class_embeddings(self):
        labels = torch.arange(self.embedding.num_embeddings).to(self.embedding.weight.device)
        return self.embedding(labels)