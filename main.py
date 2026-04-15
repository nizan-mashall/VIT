from PIL import Image
from model import processor

img = Image.open('demo_image.png')

image_processor = processor(img)
image_processor._image_scaling()
