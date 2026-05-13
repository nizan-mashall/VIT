from PIL import Image
from model import ImageProcessor
from model import PatchEmbedding
from model import TransformerEncoder
import yaml
import torch
from data_loader import CIFAR_data_loader

if __name__ == "__main__":
    # 1. Initialize the loaders
    train_loader, test_loader = CIFAR_data_loader(batch_size=32)

    # 2. Try to grab the first batch
    try:
        images, labels = next(iter(train_loader))
        
        print("✅ Success! Data loaded correctly.")
        print(f"Batch images shape: {images.shape}") # Should be [32, 3, 224, 224]
        print(f"Batch labels shape: {labels.shape}") # Should be [32]
        print(f"First 5 labels: {labels[:5]}")
        
    except Exception as e:
        print("❌ Error detected:")
        print(e)