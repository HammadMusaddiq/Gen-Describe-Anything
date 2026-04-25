# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

ip = "113.54.214.130"
port = "7890"

# os.environ['https_proxy'] = f"http://{ip}:{port}"
os.environ['https_proxy'] = ""

import argparse
import ast
import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
from dam_ import DescribeAnythingModel, disable_torch_init
import cv2


def apply_sam(image, input_points=None, input_boxes=None, input_labels=None):
    inputs = sam_processor(image, input_points=input_points, input_boxes=input_boxes,
                           input_labels=input_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0]
    scores = outputs.iou_scores[0, 0]

    mask_selection_index = scores.argmax()

    mask_np = masks[mask_selection_index].numpy()

    return mask_np


def denormalize_coordinates(coords, image_size, is_box=False):
    """Convert normalized coordinates (0-1) to pixel coordinates."""
    width, height = image_size
    if is_box:
        # For boxes: [x1, y1, x2, y2]
        x1, y1, x2, y2 = coords
        return [
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        ]
    else:
        # For points: [x, y]
        x, y = coords
        return [int(x * width), int(y * height)]


if __name__ == '__main__':
    # Example: python examples/dam_with_sam.py --image_path images/1.jpg --points '[[1172, 812], [1572, 800]]' --output_image_path output_visualization.png
    # Example: python examples/dam_with_sam.py --image_path images/1.jpg --box '[800, 500, 1800, 1000]' --use_box --output_image_path output_visualization.png
    parser = argparse.ArgumentParser(description="Describe Anything script")
    parser.add_argument('--image_dir', type=str,
                        required=True, help='image folder directory')
    parser.add_argument(
        '--points', type=str, default='[[1172, 812], [1572, 800]]', help='List of points for SAM input')
    parser.add_argument(
        '--box', type=str, default='[0, 0, 127, 255]', help='Bounding box for SAM input (x1, y1, x2, y2)')
    parser.add_argument('--use_box', action='store_true',
                        help='Use box instead of points for SAM input (default: use points)')
    parser.add_argument(
        '--query', type=str, default='<image>\nDescribe the person region in detail.', help='Prompt for the model')
    parser.add_argument('--model_path', type=str,
                        default='nvidia/DAM-3B', help='Path to the model checkpoint')
    parser.add_argument('--prompt_mode', type=str,
                        default='focal_prompt', help='Prompt mode')
    parser.add_argument('--conv_mode', type=str,
                        default='v1', help='Conversation mode')
    parser.add_argument('--temperature', type=float,
                        default=0.2, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.5,
                        help='Top-p for sampling')
    parser.add_argument('--output_image_path', type=str, default=None,
                        help='Path to save the output image with contour')
    parser.add_argument('--normalized_coords', action='store_true',
                        help='Interpret coordinates as normalized (0-1) values')
    parser.add_argument('--no_stream', action='store_true',
                        help='Disable streaming output')
    parser.add_argument('--caption_file_path', type=str, default="generated_captions.txt",
                        help='Disable streaming output')

    args = parser.parse_args()

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading SAM model...")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge", local_files_only=True).to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge", local_files_only=True)

    print("Loading DAM model...")
    disable_torch_init()
    prompt_modes = {
        "focal_prompt": "full+focal_crop",
    }
    dam = DescribeAnythingModel(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        prompt_mode=prompt_modes.get(args.prompt_mode, args.prompt_mode),
    ).to(device)


    def collect_image_paths_by_type(root_dir, image_type):
        """Return all image paths of a given type (IR, thermal, visible) from nested structure."""
        image_paths = []
        for class_id in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_id, image_type) # for created data (IR, TI, visible)
            # class_path = os.path.join(root_dir, class_id) # for only original rgb data
            if not os.path.isdir(class_path):
                continue
            for image_name in os.listdir(class_path):
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    relative_path = os.path.join(class_id, image_type, image_name) # for created data (IR, TI, visible)
                    # relative_path = os.path.join(class_id, image_name) # for only original rgb data
                    image_paths.append(relative_path)
        return image_paths

    os.makedirs('cap_predictions', exist_ok=True)

    image_type = args.caption_file_path.split("-")[-1].split(".")[0]  # 'IR', 'TI', 'visible'

    all_images = collect_image_paths_by_type(args.image_dir, image_type)

    # Load already processed captions
    existing_images = set()
    if os.path.exists(args.caption_file_path):
        with open(args.caption_file_path, 'r') as f_in:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    existing_images.add(parts[0])

    new_images = [img for img in all_images if img not in existing_images]

    print(f"Found {len(all_images)} total images, {len(existing_images)} already captioned, {len(new_images)} remaining to caption.")

    with open(args.caption_file_path, 'a') as f_out:
        for rel_path in new_images:
            try:
                image_path = os.path.join(args.image_dir, rel_path)
                print(f"\nProcessing: {rel_path}")
                img = Image.open(image_path).convert("RGB")
                image_size = img.size
                BOX = [0, 0, image_size[0], image_size[1]]

                if args.use_box:
                    input_boxes = BOX
                    if args.normalized_coords:
                        input_boxes = denormalize_coordinates(input_boxes, image_size, is_box=True)
                    input_boxes = [[input_boxes]]
                    mask_np = apply_sam(img, input_boxes=input_boxes)
                else:
                    input_points = ast.literal_eval(args.points)
                    if args.normalized_coords:
                        input_points = [denormalize_coordinates(p, image_size) for p in input_points]
                    input_labels = [1] * len(input_points)
                    input_points = [[x, y] for x, y in input_points]
                    input_points = [input_points]
                    input_labels = [input_labels]
                    mask_np = apply_sam(img, input_points=input_points, input_labels=input_labels)

                mask = Image.fromarray((mask_np * 255).astype('uint8'))

                caption = dam.get_description(
                    img,
                    mask,
                    args.query,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=1,
                    max_new_tokens=512,
                ).strip()

                f_out.write(f"{rel_path}\t{caption}\n")
                f_out.flush()

            except Exception as e:
                print(f"⚠️ Error generating caption for {rel_path}: {e}")
                continue


# python describe-anything/examples/dam_with_sam.py --image_path org_307_77_0.jpg --box '[0, 0, 127, 255]' --query '<image>\nDescribe the focused person region in detail.' --use_box --output_image_path output_visualization.png
# python describe-anything/examples/dam_with_sam.py --box '[0, 0, 127, 255]' --query '<image>\nDescribe the person in detail.' --use_box --output_image_path output_visualization.png --image_path Weixin Image_20250424201105.jpg


## MSMT17-v2 Dataset
# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi_class --use_box --image_dir "../data_folder/MSMT17_V2/mask_test_v2" --caption_file_path ../cap_predictions/MSMT17-v2-test-IR.txt ok
# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi_class --use_box --image_dir "../data_folder/MSMT17_V2/mask_test_v2" --caption_file_path ../cap_predictions/MSMT17-v2-test-TI.txt ok
# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi_class --use_box --image_dir "../data_folder/MSMT17_V2/mask_test_v2" --caption_file_path ../cap_predictions/MSMT17-v2-test-visible.txt ok

# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi_class --use_box --image_dir "../data_folder/MSMT17_V2/mask_train_v2" --caption_file_path ../cap_predictions/MSMT17-v2-train-IR.txt ok
# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi_class --use_box --image_dir "../data_folder/MSMT17_V2/mask_train_v2" --caption_file_path ../cap_predictions/MSMT17-v2-train-TI.txt ok
# CUDA_VISIBLE_DEVICES=0 python -m examples.dam_with_sam_multi_class --use_box --image_dir "../data_folder/MSMT17_V2/mask_train_v2" --caption_file_path ../cap_predictions/MSMT17-v2-train-visible.txt ok


## Real2 Dataset
# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi_class --use_box --image_dir "../data_folder/real2/real_reid_image_face_blur/images/1" --caption_file_path ../cap_predictions/real2-1-visible.txt 

# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi_class --use_box --image_dir "../data_folder/real2/real_reid_image_face_blur/images/2" --caption_file_path ../cap_predictions/real2-2-visible.txt 

# CUDA_VISIBLE_DEVICES=0 python -m examples.dam_with_sam_multi_class --use_box --image_dir "../data_folder/real2/real_reid_image_face_blur/images/3" --caption_file_path ../cap_predictions/real2-3-visible.txt 

# Dataset new path
# /mnt/data2/Hammad/Datasets/ReID_data/MSMT17_V2/mask_test_v2
# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi_class --use_box --image_dir "/mnt/data2/Hammad/Datasets/ReID_data/MSMT17_V2/mask_train_v2" --caption_file_path ../cap_predictions/MSMT17-v2-train-TI.txt
# /mnt/data2/Hammad/Datasets/ReID_data/prcc/rgb/train
# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi_class --use_box --image_dir "/mnt/data2/Hammad/Datasets/ReID_data/prcc/rgb/train" --caption_file_path ../cap_predictions/prcc-train-TI.txt
