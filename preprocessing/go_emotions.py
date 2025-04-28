# goemotions_preprocessing.py

from datasets import load_dataset
from transformers import BertTokenizer
import torch

def preprocess_goemotions(tokenizer_name='bert-base-uncased', max_len=128):
    """
    Preprocess the GoEmotions dataset.
    Args:
        tokenizer_name (str): Name of the HuggingFace tokenizer.
        max_len (int): Maximum tokenized sequence length.

    Returns:
        dict: Tokenized train and validation sets with input_ids, attention_mask, and labels.
    """
    # Load GoEmotions dataset
    dataset = load_dataset("go_emotions", "simplified")

    # Using only 'train' split for custom train/validation split
    texts = dataset['train']['text']
    labels = dataset['train']['labels']  # List of labels (we'll pick first label if multiple)

    # Simplify multiple labels to single label (pick first label if multiple)
    simplified_labels = [label[0] if len(label) > 0 else 7 for label in labels]  # 7 = 'neutral' if no label

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Tokenize texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)

    # Convert to torch tensors
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    labels_tensor = torch.tensor(simplified_labels)

    # Manual train-validation split (80-20)
    total_samples = len(input_ids)
    split_idx = int(0.8 * total_samples)

    train_dataset = {
        'input_ids': input_ids[:split_idx],
        'attention_mask': attention_mask[:split_idx],
        'labels': labels_tensor[:split_idx]
    }

    val_dataset = {
        'input_ids': input_ids[split_idx:],
        'attention_mask': attention_mask[split_idx:],
        'labels': labels_tensor[split_idx:]
    }

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_data, val_data = preprocess_goemotions()
    print("Training samples:", len(train_data['input_ids']))
    print("Validation samples:", len(val_data['input_ids']))
