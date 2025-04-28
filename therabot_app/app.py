from flask import Flask, render_template, request, jsonify, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, PreTrainedModel, RobertaConfig

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'therabot_secret_key'

# Load models
EMOTION_MODEL_PATH = r"models\Emotion_model"
SARCASM_MODEL_PATH = r"models\sarcasm_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

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

# Load generation model (fine-tuned distilGPT2)
gen_model_path = r"models\therabot-distilgpt2\checkpoint-14769"  # replace with your checkpoint
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_path).to(DEVICE)
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path)
gen_tokenizer.pad_token = gen_tokenizer.eos_token

# Load emotion and sarcasm models
#from model_utils import emotion_model, sarcasm_model, emotion_tokenizer, sarcasm_tokenizer, emotion_id2label
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

print("[INFO] Loading models...")

emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
config = RobertaConfig.from_pretrained(EMOTION_MODEL_PATH)
emotion_model = EmotionClassifier.from_pretrained(EMOTION_MODEL_PATH, config=config).to(DEVICE)
emotion_model.eval()

# Load sarcasm model
sarcasm_tokenizer = AutoTokenizer.from_pretrained(SARCASM_MODEL_PATH)
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(SARCASM_MODEL_PATH).to(DEVICE)
sarcasm_model.eval()

# Utility function to detect emotion and sarcasm
def detect_emotion_and_sarcasm(text):
    with torch.no_grad():
        # Sarcasm Detection
        enc_sarc = sarcasm_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
        logits_sarc = sarcasm_model(**enc_sarc).logits
        probs_sarc = F.softmax(logits_sarc, dim=1)
        pred_sarc = torch.argmax(probs_sarc, dim=1).item()
        sarc_conf = probs_sarc[0][pred_sarc].item()

        # Emotion Detection
        emotion_input = "[SARCASTIC] " + text if pred_sarc == 1 else text
        enc_emo = emotion_tokenizer(emotion_input, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
        logits_emo = emotion_model(**enc_emo)  # custom model returns logits directly
        probs_emo = F.softmax(logits_emo, dim=1)
        pred_emo = torch.argmax(probs_emo, dim=1).item()

        raw_emotion = emotion_id2label[pred_emo]
        adjusted_emotion = "confusion" if (pred_sarc == 1 and sarc_conf > 0.85 and raw_emotion in ["joy", "gratitude", "love"]) else raw_emotion
        sarcasm = "sarcastic" if pred_sarc == 1 else "not sarcastic"

    return adjusted_emotion, sarcasm

# Routes
@app.route('/')
def index():
    session.setdefault('history', [])
    return render_template('index.html', history=session['history'])

@app.route('/message', methods=['POST'])
def message():
    user_msg = request.json.get("text")
    emotion, sarcasm = detect_emotion_and_sarcasm(user_msg)

    # Build chat history prompt
    last_lines = session['history'][-5:]  # short-term memory
    chat_history = "\n".join(last_lines)
    prompt = f"{chat_history}\nEmotion: {emotion} | Sarcasm: {sarcasm} | User: {user_msg} | TheraBot:"

    # Generate response
    input_ids = gen_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    output_ids = gen_model.generate(input_ids, max_new_tokens=200, do_sample=True, temperature=0.7, eos_token_id=gen_tokenizer.eos_token_id)
    raw_text = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract bot's reply
    response = raw_text.split("TheraBot:")[-1].strip()
    full_entry = f"Emotion: {emotion} | Sarcasm: {sarcasm} | User: {user_msg} | TheraBot: {response}"
    session['history'].append(full_entry)

    return jsonify({"response": response, "emotion": emotion})

@app.route('/reset', methods=['POST'])
def reset():
    session['history'] = []
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(debug=True)
