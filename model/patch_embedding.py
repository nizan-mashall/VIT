import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, img_size = 224, patch_size = 16 , embedding_dim = 768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_dim))

    def forward(self, x):
        
        x = self.projection(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim = 1)
        x = x + self.positional_encoding
        return x