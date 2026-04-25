# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/
import os
import os.path as osp
from transformers import AutoConfig
from transformers import  PretrainedConfig
from huggingface_hub import snapshot_download, repo_exists
from huggingface_hub.utils import HFValidationError

def model_cached_locally(model_name):
    cache_dir = osp.expanduser("~/.cache/huggingface/hub/")
    model_folder_name = f"models--{model_name.replace('/', '--')}"
    model_dir = os.path.join(cache_dir, model_folder_name)
    return osp.exists(model_dir)

def verify_complete_model_path(model_id_or_path):
    if os.path.isdir(model_id_or_path):
        return model_id_or_path  # Already a local path

    parts = model_id_or_path.split('/')
    if len(parts) < 2:
        raise ValueError(f"Invalid model name or path: {model_id_or_path}")

    namespace, repo_name = parts[0], parts[1]
    subfolder = '/'.join(parts[2:]) if len(parts) > 2 else ''

    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    model_dir = f"models--{namespace}--{repo_name}"
    model_dir_path = os.path.join(cache_dir, model_dir)

    if not os.path.exists(model_dir_path):
        raise FileNotFoundError(f"Model cache directory does not exist: {model_dir_path}")

    snapshot_dir = os.path.join(model_dir_path, "snapshots")
    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot_dir}")

    snapshots = os.listdir(snapshot_dir)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in: {snapshot_dir}")

    snapshot = snapshots[0]  # Assume first snapshot (you can sort if needed)
    full_path = os.path.join(snapshot_dir, snapshot, subfolder)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Expected subfolder does not exist: {full_path}")

    return full_path


def get_model_config(config):
    # `mask_encoder_cfg` and `context_provider_cfg` are optional
    default_keys = ["llm_cfg", "vision_tower_cfg", "mm_projector_cfg", "mask_encoder_cfg", "context_provider_cfg"]
    
    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path
    else:
        root_path = config.resume_path 
        
    # download from huggingface
    # if root_path is not None and not osp.exists(root_path):
    #     try:
    #         valid_hf_repo = repo_exists(root_path)
    #     except HFValidationError:
    #         valid_hf_repo = False
    #     if valid_hf_repo:
    #         root_path = snapshot_download(root_path)

    if root_path is not None and not osp.exists(root_path):
        # First: check local cache
        if model_cached_locally(root_path):
            print(f"Model {root_path} already cached locally.")
            # Optionally, you could set root_path to local cache dir if needed
        else:
            # Otherwise: check online if repo exists
            try:
                valid_hf_repo = repo_exists(root_path)
            except HFValidationError:
                valid_hf_repo = False
            except Exception as e:
                print(f"Connection failed or unknown error: {e}")
                valid_hf_repo = False

            if valid_hf_repo:
                root_path = snapshot_download(root_path)
            else:
                raise RuntimeError(f"Cannot find model {root_path} locally or online.")

    return_list = []
    # for key in default_keys:
    #     cfg = getattr(config, key, None)
    #     if isinstance(cfg, dict):
    #         try:
    #             return_list.append(os.path.join(root_path, key[:-4]))
    #         except:
    #             raise ValueError(f"Cannot find resume path in config for {key}!")
    #     elif isinstance(cfg, PretrainedConfig):
    #         return_list.append(os.path.join(root_path, key[:-4]))
    #     elif isinstance(cfg, str):
    #         return_list.append(cfg)
    #     elif cfg is None:
    #         # We still return even if the cfg is None or does not exist
    #         return_list.append(cfg)

    for key in default_keys:
        # Get the configuration for the current key (model type or path)
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                resolved_path = os.path.join(root_path, key[:-4])
                resolved_path = verify_complete_model_path(resolved_path)  # Verify the complete path
                return_list.append(resolved_path)
            except Exception as e:
                raise ValueError(f"Cannot find resume path in config for {key}: {e}")
        elif isinstance(cfg, PretrainedConfig):
            resolved_path = os.path.join(root_path, key[:-4])
            resolved_path = verify_complete_model_path(resolved_path)  # Verify the complete path
            return_list.append(resolved_path)
        elif isinstance(cfg, str):
            # If the path is a string, resolve it directly
            resolved_path = verify_complete_model_path(cfg)  # Verify the complete path
            return_list.append(resolved_path)
        elif cfg is None:
            # If the model path is None, append None
            return_list.append(cfg)

    return return_list


def is_mm_model(model_path):
    """
    Check if the model at the given path is a visual language model.

    Args:
        model_path (str): The path to the model.

    Returns:
        bool: True if the model is an MM model, False otherwise.
    """
    config = AutoConfig.from_pretrained(model_path)
    architectures = config.architectures
    for architecture in architectures:
        if "llava" in architecture.lower():
            return True
    return False


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print(
            "You are using newer LLaVA code base, while the checkpoint of v0 is from older code base."
        )
        print(
            "You must upgrade the checkpoint to the new code base (this can be done automatically)."
        )
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
