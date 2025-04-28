# TheraBot

TheraBot is an AI-powered therapeutic chatbot that can detect emotions and sarcasm in user messages and provide appropriate responses. The bot uses advanced natural language processing models to understand and respond to user input in a therapeutic context.

## Features

- Real-time chat interface
- Emotion detection across 10 different emotions
- Sarcasm detection
- Context-aware responses
- Chat history management
- Responsive web interface

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended) or CPU


## Instructions

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```
## Project Structure

## Required Models

The application requires several pre-trained models to function:
- Emotion detection model (located in `models/Emotion_model/`)
- Sarcasm detection model (located in `models/sarcasm_model/`)
- Response generation model (fine-tuned DistilGPT2, located in `models/therabot-distilgpt2/`)

Make sure all model files are present in their respective directories before running the application.

## Running the Application

1. Navigate to the application directory:
```bash
cd therabot_app
```

2. Start the Flask server:
```bash
python app.py
```

3. Open your web browser and visit:
```
http://localhost:5000
```

## Usage

1. Open the web interface in your browser
2. Type your message in the chat input
3. The bot will:
   - Analyze your message for emotions and sarcasm
   - Generate an appropriate response
   - Display the response in the chat interface
4. Use the reset button to clear the chat history

## Technical Details

The application uses:
- Flask for the web server
- PyTorch for machine learning models
- Transformers library for NLP tasks
- Custom attention-based emotion classification
- Fine-tuned DistilGPT2 for response generation

## Note

The model paths in the application are currently set to absolute paths. You may need to modify these paths in `app.py` to match your system's directory structure:

- `EMOTION_MODEL_PATH`
- `SARCASM_MODEL_PATH`
- `gen_model_path`


