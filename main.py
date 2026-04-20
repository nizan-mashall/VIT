from PIL import Image
from model import processor
import yaml

with open('config.yaml','r') as file:
    config = yaml.safe_load(file)

image_path = config['paths']['image_path']
img = Image.open(image_path)

image_processor = processor(img)
image_processor._image_scaling()
