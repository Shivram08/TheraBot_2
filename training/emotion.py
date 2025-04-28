# train_emotion_classifier.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from tqdm import tqdm
import os
import json

# === Config ===
MODEL_NAME = "roberta-base"
DATA_PATH = "data/emotion_sarc_dataset.csv"
LABEL2ID_PATH = "data/label2id.json"
EPOCHS = 5
BATCH_SIZE = 32
MAX_LEN = 128
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")

# === Dataset ===
class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.texts = df["text"].tolist()
        self.labels = df["emotion"].map(label2id).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# === Model ===
class EmotionClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# === Load Data ===
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)
with open(LABEL2ID_PATH, "r") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

# === Split Data ===
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)

# === Tokenizer & DataLoader ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader = DataLoader(EmotionDataset(train_df, tokenizer, label2id), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(EmotionDataset(val_df, tokenizer, label2id), batch_size=BATCH_SIZE)

# === Model, Loss, Optimizer ===
model = EmotionClassifier(num_labels).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# === Training Loop ===
print("[INFO] Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"🔁 Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

# === Validation ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# === Results ===
print("\n Classification Report (Validation Set):")
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label)]))
print(f" Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

# === Save Model ===
save_path = "models/emotion_classifier_multitask"
os.makedirs(save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
tokenizer.save_pretrained(save_path)
print(f"\n Model saved to {save_path}")
