import numpy as np
import torch
from torchvision import transforms

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
        tensor_img = transform(self.img)
        print(tensor_img.shape)

        return
    