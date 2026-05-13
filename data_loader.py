import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def CIFAR_data_loader(batch_size = 32):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Step 1: Resize for ViT
    transforms.ToTensor(),              # Step 2: Convert to Tensor
    transforms.Normalize(               # Step 3: Normalize
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    ])

    train_dataset = datasets.CIFAR10(root = '.\data', train = True, download = False, transform = transform)
    test_dataset = datasets.CIFAR10(root = '.\data', train = False, download = False, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader
