# 🧠 TheraBot
**An AI-Powered Therapeutic Chatbot with Emotion and Sarcasm Detection**

TheraBot is an AI-driven virtual therapist that detects emotions and sarcasm in user messages and generates context-aware, empathetic responses in real time.

---

## Table of Contents
- [Features](#features)
- [Application Interface](#application-interface)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Required Models](#required-models)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Configuration Notes](#configuration-notes)
- [License & Disclaimer](#license--disclaimer)

---

## Features
- 💬 **Real-time chat interface**
- 😌 **Emotion detection** across up to 10 emotions
- 😏 **Sarcasm detection**
- 🧠 **Context-aware responses** (fine-tuned DistilGPT2)
- 🕒 **Chat history management**
- 📱 **Responsive web interface**

---

## Application Interface

> Place these images in the same folder as `README.md` (or update the paths).

### Example 1 — Emotional Support Interaction  
![TheraBot UI 1](./bd02dff9-e01b-4a7b-8cd6-c56fb2723365.png)

### Example 2 — Empathetic Conversation  
![TheraBot UI 2](./6d2ce302-766d-4492-9379-dbf05791d159.png)

---

## Prerequisites
- Python **3.7+**
- **CUDA-compatible GPU** (recommended) or CPU

---

## Setup

### 1) Create and activate a virtual environment
\`\`\`bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
\`\`\`

### 2) Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## Project Structure
\`\`\`
therabot/
├─ models/
│  ├─ Emotion_model/
│  ├─ sarcasm_model/
│  └─ therabot-distilgpt2/
├─ static/              # CSS, images, etc.
├─ templates/
│  └─ index.html
├─ app.py
├─ requirements.txt
└─ README.md
\`\`\`

---

## Required Models
The application requires several pre-trained models:

| Model | Directory |
| --- | --- |
| Emotion detection | `models/Emotion_model/` |
| Sarcasm detection | `models/sarcasm_model/` |
| Response generation (fine-tuned DistilGPT2) | `models/therabot-distilgpt2/` |

Ensure all model files are present in their respective directories before running.

---

## Running the Application

> If your entrypoint lives in a subfolder, `cd` there first (e.g., `cd therabot_app`).

\`\`\`bash
python app.py
\`\`\`

Then open:
\`\`\`
http://127.0.0.1:5000
\`\`\`

---

## Usage
1. Open the web interface in your browser.  
2. Type a message in the chat input.  
3. TheraBot will:
   - analyze your message for **emotions** and **sarcasm**,
   - generate a supportive response, and
   - display it instantly.  
4. Use **Reset** to clear chat history.

---

## Technical Details
- **Framework:** Flask  
- **DL/NLP:** PyTorch, Hugging Face Transformers  
- **Emotion Model:** Custom attention-based classifier  
- **Generator:** Fine-tuned DistilGPT2

---

## Configuration Notes
Some model paths may be configured as absolute paths in `app.py`. Update them to match your environment:
\`\`\`python
EMOTION_MODEL_PATH = "path/to/models/Emotion_model"
SARCASM_MODEL_PATH = "path/to/models/sarcasm_model"
GEN_MODEL_PATH     = "path/to/models/therabot-distilgpt2"
\`\`\`

---

## License & Disclaimer
TheraBot is intended for educational and research purposes and is **not a substitute for professional mental health services**. Use responsibly.
