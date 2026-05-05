import torch
import torch.nn as nn
from model import TransformerEncoder
from model import ImageProcessor
from model import PatchEmbedding

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_processor
        self.patchembedding
        self.transformerencoder

        