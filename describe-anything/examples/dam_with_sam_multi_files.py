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


def add_contour(img, mask, input_points=None, input_boxes=None):
    img = img.copy()

    # Draw contour
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=6)

    # Draw points if provided
    if input_points is not None:
        for points in input_points:  # Handle batch of points
            for x, y in points:
                # Draw a filled circle for each point
                cv2.circle(img, (int(x), int(y)), radius=10,
                           color=(1.0, 0.0, 0.0), thickness=-1)
                # Draw a white border around the circle
                cv2.circle(img, (int(x), int(y)), radius=10,
                           color=(1.0, 1.0, 1.0), thickness=2)

    # Draw boxes if provided
    if input_boxes is not None:
        for box_batch in input_boxes:  # Handle batch of boxes
            for box in box_batch:  # Iterate through boxes in the batch
                x1, y1, x2, y2 = map(int, box)
                # Draw rectangle with white color
                cv2.rectangle(img, (x1, y1), (x2, y2),
                              color=(1.0, 1.0, 1.0), thickness=4)
                # Draw inner rectangle with red color
                cv2.rectangle(img, (x1, y1), (x2, y2),
                              color=(1.0, 0.0, 0.0), thickness=2)

    return img


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


def print_streaming(text):
    """Helper function to print streaming text with flush"""
    print(text, end="", flush=True)


if __name__ == '__main__':
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
    

    # Load already processed captions
    existing_images = set()
    if os.path.exists(args.caption_file_path):
        with open(args.caption_file_path, 'r') as f_in:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    existing_images.add(parts[0])

    # Collect image files
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]

    # Filter only images not already processed
    new_images = [f for f in image_files if f not in existing_images]

    print(f"Found {len(image_files)} total images, {len(existing_images)} already captioned, {len(new_images)} remaining to caption.")

    os.makedirs('cap_predictions', exist_ok=True)

    # Open file in append mode
    with open(args.caption_file_path, 'a') as f_out:
    # with open(args.caption_file_path, 'w') as f_out:
        for filename in new_images:
            try:
                image_path = os.path.join(args.image_dir, filename)
                print(f"\nProcessing: {filename}")
                img = Image.open(image_path).convert("RGB")
                image_size = img.size  # (width, height)
                BOX = [0, 0, image_size[0], image_size[1]]

                if args.use_box:
                    # input_boxes = ast.literal_eval(args.box)
                    input_boxes = BOX
                    if args.normalized_coords:
                        input_boxes = denormalize_coordinates(
                            input_boxes, image_size, is_box=True)
                    input_boxes = [[input_boxes]]  # Add an extra level of nesting
                    # print(f"Using input_boxes: {input_boxes}")
                    mask_np = apply_sam(img, input_boxes=input_boxes)
                else:
                    input_points = ast.literal_eval(args.points)
                    if args.normalized_coords:
                        input_points = [denormalize_coordinates(point, image_size)
                                        for point in input_points]
                    # Assume all points are foreground
                    input_labels = [1] * len(input_points)
                    input_points = [[x, y]
                                    for x, y in input_points]  # Convert to list of lists
                    input_points = [input_points]  # Wrap in outer list
                    input_labels = [input_labels]  # Wrap labels in list
                    # print(f"Using input_points: {input_points}")
                    mask_np = apply_sam(img, input_points=input_points,
                                        input_labels=input_labels)

                mask = Image.fromarray((mask_np * 255).astype(np.uint8))

                # Generate caption (non-streaming)
                caption = dam.get_description(
                    img,
                    mask,
                    args.query,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=1,
                    max_new_tokens=512,
                ).strip()

                # print(f"→ Caption: {caption}")
                f_out.write(f"{filename}\t{caption}\n")
                f_out.flush()  # Ensure the output is written immediately

            except Exception as e:
                print(f"⚠️ Error generating caption for {filename}: {e}")
                continue

    # Save the mask image
    # if args.output_image_path:
    #     img_np = np.asarray(img).astype(float) / 255.0

    #     # Prepare visualization inputs
    #     vis_points = input_points if not args.use_box else None
    #     vis_boxes = input_boxes if args.use_box else None

    #     img_with_contour_np = add_contour(img_np, mask_np,
    #                                       input_points=vis_points,
    #                                       input_boxes=vis_boxes)
    #     img_with_contour_pil = Image.fromarray(
    #         (img_with_contour_np * 255.0).astype(np.uint8))
    #     img_with_contour_pil.save(args.output_image_path)
    #     print(f"Output image with contour saved as {args.output_image_path}")


# python describe-anything/examples/dam_with_sam.py --image_path org_307_77_0.jpg --box '[0, 0, 127, 255]' --query '<image>\nDescribe the focused person region in detail.' --use_box --output_image_path output_visualization.png
# python describe-anything/examples/dam_with_sam.py --box '[0, 0, 127, 255]' --query '<image>\nDescribe the person in detail.' --use_box --output_image_path output_visualization.png --image_path Weixin Image_20250424201105.jpg



# python describe-anything/examples/dam_with_sam_multi.py --use_box --image_dir ./ --caption_file_path ./generated_captions.txt

### Celeb-reID Dataset
# python describe-anything/examples/dam_with_sam_multi.py --use_box --image_dir "data_folder/Celeb-reID/gallery/IR" --caption_file_path cap_predictions/Celeb-reID-gallery-IR.txt ok
# CUDA_VISIBLE_DEVICES=0 python -m examples.dam_with_sam_multi --use_box --image_dir "../data_folder/Celeb-reID/gallery/TI" --caption_file_path ../cap_predictions/Celeb-reID-gallery-TI.txt ok
# CUDA_VISIBLE_DEVICES=0 python -m examples.dam_with_sam_multi --use_box --image_dir "../data_folder/Celeb-reID/gallery/visible" --caption_file_path ../cap_predictions/Celeb-reID-gallery-visible.txt ok

# CUDA_VISIBLE_DEVICES=0 python -m examples.dam_with_sam_multi --use_box --image_dir "../data_folder/Celeb-reID/query/IR" --caption_file_path ../cap_predictions/Celeb-reID-query-IR.txt ok
# CUDA_VISIBLE_DEVICES=0 python -m examples.dam_with_sam_multi --use_box --image_dir "../data_folder/Celeb-reID/query/TI" --caption_file_path ../cap_predictions/Celeb-reID-query-TI.txt ok
# CUDA_VISIBLE_DEVICES=0 python -m examples.dam_with_sam_multi --use_box --image_dir "../data_folder/Celeb-reID/query/visible" --caption_file_path ../cap_predictions/Celeb-reID-query-visible.txt ok

# CUDA_VISIBLE_DEVICES=0 python -m examples.dam_with_sam_multi --use_box --image_dir "../data_folder/Celeb-reID/train/IR" --caption_file_path ../cap_predictions/Celeb-reID-train-IR.txt ok
# CUDA_VISIBLE_DEVICES=2 python -m examples.dam_with_sam_multi --use_box --image_dir "../data_folder/Celeb-reID/train/TI" --caption_file_path ../cap_predictions/Celeb-reID-train-TI.txt ok
# CUDA_VISIBLE_DEVICES=0 python -m examples.dam_with_sam_multi --use_box --image_dir "../data_folder/Celeb-reID/train/visible" --caption_file_path ../cap_predictions/Celeb-reID-train-visible.txt ok

# RGBNT201
# CUDA_VISIBLE_DEVICES=1 python -m examples.dam_with_sam_multi_files --use_box --image_dir "/mnt/data2/Hammad/Datasets/ReID_data/RGBNT201/train_141/RGB" --caption_file_path ../cap_predictions/RGBNT201-train-visible.txt


