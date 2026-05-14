"""
train_image_efficientnet.py
----------------------------
Fine-tunes EfficientNet-B3 for phishing webpage screenshot detection.

Your dataset folder should look like:
  image_dataset_new/
  ├── train/
  │   ├── legitimate/   ← legit site screenshots
  │   └── phishing/     ← phishing site screenshots
  └── val/
      ├── legitimate/
      └── phishing/

Install dependencies:
  pip install torch torchvision
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
import numpy as np
import os

# ─────────────────── CONFIG ───────────────────
DATA_DIR       = "image_dataset_new"
MODEL_SAVE     = "image_model_efficientnet.pth"
BATCH_SIZE     = 16
EPOCHS         = 10
LR             = 1e-4
SEED           = 42
IMG_SIZE       = 300        # EfficientNet-B3 native resolution
# ──────────────────────────────────────────────

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────── TRANSFORMS ───────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────── DATASETS ───────────────
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

classes = train_dataset.classes
print(f"Classes: {classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ─────────────── MODEL ───────────────
print("Loading EfficientNet-B3...")
model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

# Replace classifier head for 2 classes
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, 2)
)

model = model.to(device)

# ─────────────── LOSS / OPTIM ───────────────
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ─────────────── TRAINING ───────────────
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out   = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    return correct / total, all_preds, all_labels

best_val_acc = 0.0
print("\nStarting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, (imgs, lbls) in enumerate(train_loader):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    val_acc, preds, labels = evaluate(model, val_loader)
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"  ✓ Best model saved (val acc: {val_acc*100:.2f}%)")

print(f"\nTraining complete! Best val accuracy: {best_val_acc*100:.2f}%")
print(f"Model saved to '{MODEL_SAVE}'")

# Final report
val_acc, preds, labels = evaluate(model, val_loader)
print("\nFinal Classification Report:")
print(classification_report(labels, preds, target_names=classes))
