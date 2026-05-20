import torch
import torch.nn as nn
from model import TransformerEncoder
from model import PatchEmbedding

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10,
                 depth=6, embedding_dim=768, n_heads=8, dropout_rate=0.1, hidden_dim=2048):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(
            in_channels = in_channels,
            img_size = img_size,
            patch_size = patch_size,
            embedding_dim = embedding_dim)
        
        self.transformer_encoder = TransformerEncoder(
            depth = depth, 
            dim = embedding_dim , 
            n_heads = n_heads, 
            dropout_rate = dropout_rate, 
            hidden_dim = hidden_dim)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        cls_token_output = x[:, 0, :]
        logits = self.mlp_head(cls_token_output)
        return logits