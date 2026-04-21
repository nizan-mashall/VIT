from PIL import Image
from model import ImageProcessor
import yaml
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

with open('config.yaml','r') as file:
    config = yaml.safe_load(file)

IMG_PATH = config['paths']['image_path']
IMG_HEIGHT = config['processor']['image_height']
IMG_WIDTH = config['processor']['image_width']
PATCH_SIZE = config['processor']['patch_size']

print(f"Opening the image: {IMG_PATH}, in resulotion of: ({IMG_HEIGHT}/{IMG_WIDTH})")

img = Image.open(IMG_PATH).convert('RGB')

image_processor = ImageProcessor(img, IMG_HEIGHT, IMG_WIDTH, PATCH_SIZE)
image_processor._image_scaling()
image_processor._split_to_patches()

