import torch
import torch.nn as nn
from model import TransformerEncoder
from model import ImageProcessor
from model import PatchEmbedding

class VisionTransformer(nn.Module):
    def __init__(self, img, h, w, patch_size, batch_size, depth, dim , n_heads, dropout_rate, hidden_dim):
        super().__init__()
        self.image_processor = ImageProcessor(img, h, w, patch_size)
        self.patch_embedding = PatchEmbedding(img, h, w, patch_size, batch_size)
        self.transformer_encoder = TransformerEncoder(depth, dim , n_heads, dropout_rate, hidden_dim)

    def forward(self, x):
        x = self.image_processor()
        x = self.patch_embedding(x)
pixel2embedding = PatchEmbedding(img_tensor, IMG_HEIGHT, IMG_WIDTH, PATCH_SIZE, BATCH_SIZE)
img_embedding = pixel2embedding(img_tensor)

