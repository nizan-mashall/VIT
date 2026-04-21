import numpy as np
import torch
from torchvision import transforms
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as f

class ImageProcessor:
    def __init__(self, img, h, w, patch_size):
        self.img = img
        self.h = h
        self.w = w
        self.patch_size = patch_size

   
    def _image_scaling(self):
        self.img.show()
        self.img = self.img.resize((self.h,self.w))
        return self.img
    
    def _split_to_patches(self):
        transform = transforms.ToTensor()
        tensor_img = transform(self.img).unsqueeze(0)   # unsqueeze adding the batch dim
        #print(tensor_img.shape)
        tensor_img = rearrange(tensor_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1 = self.patch_size, p2 = self.patch_size)
        #print(tensor_img.shape)
        return tensor_img
    
    def image_Processing(self):
        self.img = self._image_scaling()
        return self._split_to_patches()
    