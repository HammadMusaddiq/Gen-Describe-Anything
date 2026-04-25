import os

ip = "113.54.162.196"
port = "7890"

os.environ['https_proxy'] = f"http://{ip}:{port}"

from huggingface_hub import snapshot_download
from huggingface_hub import login

# Use your Hugging Face access token to log in

import os

HF_TOKEN = os.getenv("HF_TOKEN")

login(token=HF_TOKEN)

# Replace this with your model repo
# repo_id = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
# repo_id = "meta-llama/Llama-3.2-1B"
# repo_id = "google/gemma-3-4b-it"
# repo_id = "google/txgemma-2b-predict"
repo_id = "meta-llama/Llama-3.2-3B-Instruct"

# Download all files to a local directory
local_dir = snapshot_download(
    repo_id=repo_id,
    cache_dir="./hf_models",
    local_dir="/mnt/database/hammad/hf_models/Llama-3.2-3B-Instruct",
    )

print(f"Model downloaded to: {local_dir}")
