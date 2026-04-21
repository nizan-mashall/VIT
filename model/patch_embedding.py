import torch
import torch.nn as nn
class PatchEmbedding(nn.Module):
    def __init__(self, img, h, w, patch_size):
        super().__init__()
        self.h = h
        self.w = w
        self.patch_size = patch_size
        input_dim = 3 * patch_size**2
        embedding_dim = 768
        self.projection = nn.Linear(input_dim, embedding_dim)

    def forward(self,img):
        img = self.projection(img)
        return img