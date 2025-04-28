import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, PreTrainedModel, RobertaConfig

# ------------------ CONFIG ------------------
EMOTION_MODEL_PATH = r"models\Emotion_model"
SARCASM_MODEL_PATH = r"models\sarcasm_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Emotion ID to Label
emotion_id2label = {
    0: "anger",
    1: "confusion",
    2: "desire",
    3: "disgust",
    4: "gratitude",
    5: "joy",
    6: "love",
    7: "neutral",
    8: "sadness",
    9: "surprise"
}

# ------------------ EMOTION MODEL CLASS ------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)

class EmotionClassifier(PreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained("vinai/bertweet-base")
        hidden_dim = self.encoder.config.hidden_size
        self.attn = AttentionPooling(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = self.attn(outputs.last_hidden_state, attention_mask)
        x = self.dropout(x)
        return self.classifier(x)

# ------------------ LOAD MODELS ------------------
print("[INFO] Loading models...")

# Load tokenizer + emotion model
emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
config = RobertaConfig.from_pretrained(EMOTION_MODEL_PATH)
emotion_model = EmotionClassifier.from_pretrained(EMOTION_MODEL_PATH, config=config).to(DEVICE)
emotion_model.eval()

# Load sarcasm model
sarcasm_tokenizer = AutoTokenizer.from_pretrained(SARCASM_MODEL_PATH)
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(SARCASM_MODEL_PATH).to(DEVICE)
sarcasm_model.eval()

# ------------------ INFERENCE FUNCTION ------------------
def run_joint_inference(text):
    # Emotion Inference
    enc = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = emotion_model(**enc)
        probs = F.softmax(logits, dim=1)
        pred_emotion = torch.argmax(probs, dim=1).item()
        emotion_conf = probs[0][pred_emotion].item()

    # Sarcasm Inference
    enc_sarc = sarcasm_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits_sarc = sarcasm_model(**enc_sarc).logits
        probs_sarc = F.softmax(logits_sarc, dim=1)
        pred_sarc = torch.argmax(probs_sarc, dim=1).item()
        sarc_conf = probs_sarc[0][pred_sarc].item()

    # ------------------ Heuristic Correction ------------------
    raw_emotion = emotion_id2label[pred_emotion]
    adjusted_emotion = raw_emotion
    note = ""

    if raw_emotion in ["joy", "love", "gratitude"] and pred_sarc == 1:
        adjusted_emotion = "confusion"
        note = "🌀 Emotion adjusted due to sarcasm"

    return {
        "input": text,
        "emotion": adjusted_emotion,
        "emotion_confidence": round(emotion_conf, 3),
        "sarcasm": "sarcastic" if pred_sarc == 1 else "not sarcastic",
        "sarcasm_confidence": round(sarc_conf, 3),
        "note": note
    }

# ------------------ CLI LOOP ------------------
if __name__ == "__main__":
    while True:
        text = input("Enter a sentence to analyze (or type 'exit'): ")
        if text.lower() == "exit":
            break
        result = run_joint_inference(text)
        print("\n Joint Inference Result:")
        for k, v in result.items():
            if k != "note":
                print(f"{k}: {v}")
        if result.get("note"):
            print(result["note"])
        print()
