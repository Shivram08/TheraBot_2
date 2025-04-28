# train_sarcasm_classifier.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

class SarcasmClassifier(nn.Module):
    def __init__(self, model_name):
        super(SarcasmClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 2)  # Binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(pooled_output)
        return logits

def load_data_from_csv(csv_path, tokenizer_name='roberta-base', max_len=128):
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_len)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_len)

    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(train_labels)
    )

    val_dataset = TensorDataset(
        torch.tensor(val_encodings['input_ids']),
        torch.tensor(val_encodings['attention_mask']),
        torch.tensor(val_labels)
    )

    return train_dataset, val_dataset

def train(model, train_loader, val_loader, device, epochs=5, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_sarcasm_model.pt")
            print("Saved Best Model!\n")

def evaluate(model, val_loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, label_ids = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.cpu().numpy())
            labels.extend(label_ids.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    csv_path = "path_to_sarc_dataset.csv"  # Update this
    train_dataset, val_dataset = load_data_from_csv(csv_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SarcasmClassifier(model_name="roberta-base")
    model.to(device)

    train(model, train_loader, val_loader, device, epochs=5)

    print("Training Completed!")
