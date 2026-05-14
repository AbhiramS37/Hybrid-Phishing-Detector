"""
train_url_bert.py
-----------------
Fine-tunes bert-base-uncased for phishing URL detection.

Best dataset: "malicious_phish.csv" from Kaggle
Link: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

Download it and place malicious_phish.csv in the same folder as this script.

The CSV has two columns:
  url   → the raw URL string
  type  → benign / phishing / malware / defacement

We treat:
  benign      → legitimate (label 1)
  everything else → phishing (label 0)

Install dependencies first:
  pip install transformers datasets scikit-learn torch torchvision
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# ─────────────────── CONFIG ───────────────────
CSV_PATH       = "malicious_phish.csv"
MODEL_SAVE     = "url_bert_model"          # folder where model is saved
MAX_LEN        = 128                       # max token length for a URL
BATCH_SIZE     = 32
EPOCHS         = 3
LR             = 2e-5
SEED           = 42
# ──────────────────────────────────────────────

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────── LOAD & PREP DATA ───────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()

# Keep only url and type columns
df = df[["url", "type"]].dropna()

# Binary label: 1 = legitimate, 0 = phishing/malicious
df["label"] = (df["type"].str.strip().str.lower() == "benign").astype(int)

# Balance classes (cap at 100k each so training is fast)
legit   = df[df["label"] == 1].sample(min(100_000, (df["label"]==1).sum()), random_state=SEED)
phish   = df[df["label"] == 0].sample(min(100_000, (df["label"]==0).sum()), random_state=SEED)
df      = pd.concat([legit, phish]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Dataset size: {len(df)} | Legit: {(df.label==1).sum()} | Phishing: {(df.label==0).sum()}")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df["label"])

# ─────────────── TOKENIZER ───────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, max_len):
        self.urls      = urls.reset_index(drop=True)
        self.labels    = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.urls[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = URLDataset(train_df["url"], train_df["label"], tokenizer, MAX_LEN)
val_dataset   = URLDataset(val_df["url"],   val_df["label"],   tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─────────────── MODEL ───────────────
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                             num_warmup_steps=total_steps // 10,
                                             num_training_steps=total_steps)

# ─────────────── TRAINING ───────────────
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["label"].to(device)
            out   = model(ids, attention_mask=mask)
            preds = out.logits.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    acc = correct / total
    return acc, all_preds, all_labels

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["label"].to(device)

        optimizer.zero_grad()
        out  = model(ids, attention_mask=mask, labels=lbls)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}")

    val_acc, preds, labels = evaluate(model, val_loader)
    print(f"\nEpoch {epoch+1} complete | Val Accuracy: {val_acc*100:.2f}%")
    print(classification_report(labels, preds, target_names=["phishing", "legitimate"]))

# ─────────────── SAVE ───────────────
os.makedirs(MODEL_SAVE, exist_ok=True)
model.save_pretrained(MODEL_SAVE)
tokenizer.save_pretrained(MODEL_SAVE)
print(f"\nModel saved to '{MODEL_SAVE}/'")
print("Done! Use this folder path in app.py")
