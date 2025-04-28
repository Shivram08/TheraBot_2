import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import csv
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, PreTrainedModel, RobertaConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, PreTrainedModel, RobertaConfig
import torch.nn as nn

# ========== Paths ==========
EMOTION_MODEL_PATH = "E:\NLP_Project_final\models\Emotion_model"
SARCASM_MODEL_PATH = "E:\NLP_Project_final\models\sarcasm_model"
OUTPUT_FILE = "psychology10k_empathy.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== Emotion ID Map ==========
emotion_id2label = {
    0: "anger", 1: "confusion", 2: "desire", 3: "disgust", 4: "gratitude",
    5: "joy", 6: "love", 7: "neutral", 8: "sadness", 9: "surprise"
}


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

# ========== Load Models ==========
print(f"[INFO] Using device: {DEVICE}")
print("[INFO] Loading models...")
emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
config = RobertaConfig.from_pretrained(EMOTION_MODEL_PATH)
emotion_model = EmotionClassifier.from_pretrained(EMOTION_MODEL_PATH, config=config).to(DEVICE)
emotion_model.eval()

sarcasm_tokenizer = AutoTokenizer.from_pretrained(SARCASM_MODEL_PATH)
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(SARCASM_MODEL_PATH).to(DEVICE)
sarcasm_model.eval()

# ========== Inference Function ==========
def run_joint_inference(text):
    # Sarcasm detection
    enc_sarc = sarcasm_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits_sarc = sarcasm_model(**enc_sarc).logits
        probs_sarc = F.softmax(logits_sarc, dim=1)
        pred_sarc = torch.argmax(probs_sarc, dim=1).item()
        sarc_conf = probs_sarc[0][pred_sarc].item()

    # Emotion detection (with sarcasm prepending)
    emotion_input = "[SARCASTIC] " + text if pred_sarc == 1 else text
    enc = emotion_tokenizer(emotion_input, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        logits = emotion_model(**enc)
        probs = F.softmax(logits, dim=1)
        pred_emotion = torch.argmax(probs, dim=1).item()
        emotion_conf = probs[0][pred_emotion].item()

    # Adjustment
    raw_emotion = emotion_id2label[pred_emotion]
    adjusted_emotion = raw_emotion
    if sarc_conf > 0.85 and sarc_conf < 1:
        if raw_emotion in ["joy", "gratitude", "love"]:
            adjusted_emotion = "anger"
    if sarc_conf > 0.55 and sarc_conf < 0.85:
        if raw_emotion in ["joy", "gratitude", "love"]:
            adjusted_emotion = "sadness"
    if sarc_conf < 0.55 and sarc_conf > 0.25:
        if raw_emotion in ["joy", "gratitude", "love"]:
            adjusted_emotion = "confusion"

    return {
        "emotion": adjusted_emotion,
        "sarcasm": "sarcastic" if pred_sarc == 1 else "not sarcastic"
    }

# ========== Load Dataset ==========

# ========== Load Dataset ==========
print("[INFO] Loading Hugging Face dataset...")
dataset = load_dataset("samhog/psychology-10k")
data_split = dataset["train"]  # change to "test" or "validation" if needed

# ------------------ Process & Save ------------------
print("[INFO] Starting processing...")
with open(OUTPUT_FILE, mode="w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["prompt", "response"])  # CSV header

    for i, row in enumerate(data_split):
        try:
            user = row["input"].strip().replace("\n", " ")
            response = row["output"].strip().replace("\n", " ")

            result = run_joint_inference(user)

            prompt = (
                f"Emotion: {result['emotion']} | Sarcasm: {result['sarcasm']} | "
                f"User: {user} | TheraBot:"
            )

            writer.writerow([prompt, response])

            if i % 100 == 0:
                print(f"[{datetime.now()}] Processed {i}/{len(data_split)}")

        except Exception as e:
            print(f"[ERROR] Skipped row {i}: {e}")
            continue

print(f" Dataset saved as: {OUTPUT_FILE}")

