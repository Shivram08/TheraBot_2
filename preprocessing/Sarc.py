# sarc_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
import torch

def preprocess_sarc(data_path, tokenizer_name='roberta-base', max_len=128):
   
    # Load dataset
    df = pd.read_csv(data_path) # download SARC from kaggle

    # Expected columns: 'text', 'label'
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # Train-validation split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

    # Tokenize
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_len)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_len)

    # Convert to torch tensors
    train_dataset = {
        'input_ids': torch.tensor(train_encodings['input_ids']),
        'attention_mask': torch.tensor(train_encodings['attention_mask']),
        'labels': torch.tensor(train_labels)
    }

    val_dataset = {
        'input_ids': torch.tensor(val_encodings['input_ids']),
        'attention_mask': torch.tensor(val_encodings['attention_mask']),
        'labels': torch.tensor(val_labels)
    }

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_data, val_data = preprocess_sarc("path_to_sarc_dataset.csv")
    print("Training samples:", len(train_data['input_ids']))
    print("Validation samples:", len(val_data['input_ids']))
