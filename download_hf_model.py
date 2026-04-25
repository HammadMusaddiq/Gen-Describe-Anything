import os

ip = "113.54.162.196"
port = "7890"

os.environ['https_proxy'] = f"http://{ip}:{port}"
# export https_proxy=http://113.54.162.196:7890

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_api

# Model name
# model_name = "google/gemma-3-4b-it"
# model_name = "google/txgemma-2b-predict"
model_name = "meta-llama/Llama-3.2-1B"

# Attempt to download the model
try:
    print(f"Downloading model: {model_name}...")

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Model '{model_name}' downloaded and loaded successfully.")
except Exception as e:
    print(f"Error downloading model: {e}")

# import requests

# try:
#     response = requests.get("https://huggingface.co")
#     print(response.status_code)
# except requests.exceptions.RequestException as e:
#     print(f"Error: {e}")
