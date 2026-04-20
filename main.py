from PIL import Image
from model import ImageProcessor
import yaml
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

with open('config.yaml','r') as file:
    config = yaml.safe_load(file)

image_path = config['paths']['image_path']
image_height = config['processor']['image_height']
image_width = config['processor']['image_width']
patch_size = config['processor']['patch_size']

print(f"Opening the image: {image_path}, in resulotion of: ({image_height}/{image_width})")

img = Image.open(image_path).convert('RGB')

image_processor = ImageProcessor(img, image_height, image_width, patch_size)
image_processor._image_scaling()
image_processor._split_to_patches()

