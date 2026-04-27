from PIL import Image
from model import ImageProcessor
from model import PatchEmbedding
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
BATCH_SIZE = config['training']['batch_size']

print(f"Opening the image: {IMG_PATH}, in resulotion of: ({IMG_HEIGHT}/{IMG_WIDTH})")

img = Image.open(IMG_PATH).convert('RGB')

image_processor = ImageProcessor(img, IMG_HEIGHT, IMG_WIDTH, PATCH_SIZE)
img_tensor = image_processor.image_Processing()
pixel2embedding = PatchEmbedding(img_tensor, IMG_HEIGHT, IMG_WIDTH, PATCH_SIZE, BATCH_SIZE)
img_embedding = pixel2embedding(img_tensor)
