from flask import request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model dan tokenizer sekali saat modul diimpor
base_model = "model/chatbot/"
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map='cpu',
)

def generate_response(inputs):
    outputs = model.generate(
        **inputs,
        max_length=75,
        num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response.split(".")[0].strip()
    return response_text

def chat():
    data = request.get_json()
    prompt = data.get('prompt')
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    response_text = generate_response(inputs)
    jawaban = {
        'response': response_text
    }
    return jsonify(jawaban), 200



