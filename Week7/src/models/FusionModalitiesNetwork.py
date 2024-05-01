import torch
import torch.nn as nn

from .EarlyFusion import EmbeddingFusion


class FusionModalitiesNet(nn.Module):
    def __init__(self, video_model: nn.Module, embedding_size: int = 512, num_classes: int = 51) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        
        self.video_model = video_model
        self.vector_fusion = EmbeddingFusion(self.embedding_size)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, x, y): # Forward pass requires video input of shape [B, C, T, H, W] and skeleton embeddings [B, 512]
        x = self.video_model(x)
        x = self.vector_fusion(x, y)
        x = self.classifier(x)
        return x