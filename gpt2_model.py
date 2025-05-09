from flask import jsonify
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import speech_recognition as sr

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

# Speech-to-text function
def convert_speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Speech recognition service is unavailable."

# Text-to-text function using GPT-2
def generate_text(prompt):
    numeric_ids = tokenizer.encode(prompt, return_tensors='pt')
    result = model.generate(
        numeric_ids,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
    return generated_text
