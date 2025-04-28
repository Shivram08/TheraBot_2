import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
model_path = r"models\therabot-distilgpt2\checkpoint-14769"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Assign safe pad token

model = AutoModelForCausalLM.from_pretrained(model_path)
model.config.pad_token_id = tokenizer.pad_token_id

model.to("cpu")  # move to CPU first
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU inference

prompt = "Emotion: sadness | Sarcasm: not sarcastic | User: I feel like nothing I do is ever enough.. | TheraBot:"

output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]['generated_text']
print(output)
