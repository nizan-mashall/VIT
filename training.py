from PIL import Image
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import CIFAR_data_loader
from model import VisionTransformer
import os

CHECKPOINT_DIR = '/code/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train(BATCH_SIZE = 32, EPOCHS = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running device: {device}")
    train_loader, test_loader = CIFAR_data_loader(batch_size=BATCH_SIZE)

    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=10,
        depth=6,
        embedding_dim=768,
        n_heads=8,
        dropout_rate=0.1,
        hidden_dim=2048
    ).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"--- Epoch {epoch+1} Finished | Avg Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% ---")

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"--- Validation Acc: {val_acc:.2f}% ---")

        # ---- Save best model ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/best_model.pth')
            print(f"Best model saved! (Val Acc: {val_acc:.2f}%)")

if __name__ == "__main__":
    train()