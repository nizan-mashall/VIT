import torch
import torch.nn as nn
class PatchEmbedding(nn.Module):
    def __init__(self, img, h, w, patch_size, batch_size):
        super().__init__()
        self.h = h
        self.w = w
        self.patch_size = patch_size
        embedding_dim = 768
        self.batch_size = batch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = nn.Parameter(torch.randn(1, 197, 768))
        input_dim = 3 * patch_size**2
        self.projection = nn.Linear(input_dim, embedding_dim)

    def forward(self,img):
        
        img = self.projection(img)
        current_batch_size = img.shape[0]
        cls_token = self.cls_token.expand(current_batch_size, -1, -1)
        positional_encoding = self.positional_encoding.expand(current_batch_size, -1, -1)
        img = torch.cat([cls_token, img], dim = 1)
        img = img + positional_encoding
        return img