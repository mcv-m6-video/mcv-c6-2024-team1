import torch
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class LearnableVec(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(n))
  
    def forward(self, x):
        return self.W * x


class EmbeddingFusion(nn.Module):
    def __init__(self, num_vecs: int = 2, embedding_size: int = 512) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.weight_vecs = nn.ModuleList([LearnableVec(embedding_size) for i in range(num_vecs)])
        
    
    def forward(self, *x): # n vectors of shape [B, embedding_size]
        res = torch.zeros(x[0].shape[0], self.embedding_size).to(device)
        for i in range(len(x)):
            res += self.weight_vecs[i](x[i])
        return res