# import os
# import time
# from PIL import Image, UnidentifiedImageError
# from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
# import torch

# # === SETTINGS ===
# IMAGE_DIR = "./"
# SAVE_PATH = "./cleaned_captions.txt"

# # === PROMPTS ===
# GEN_PROMPT = (
#     "Write a short descriptive caption for this image, include features like what the person in the object "
#     "have wear on upper body as t-shirt, shirt, coat, and lower body as trouser, pant, and what he wears, like "
#     "joggers, sleepers, and what he has on his head, a cap etc, and her hair colour if visible, age, ethnicity."
# )
# # CLEAN_PROMPT_TEMPLATE = (
# #     "You are a caption refinement assistant.\n"
# #     "Your job is to clean and shorten noisy image captions for person re-identification tasks.\n"
# #     "Keep only essential visual details related to the main person in the image.\n"
# #     "Remove background info, irrelevant objects, or vague fluff.\n"
# #     "Input caption: \"{}\"\n"
# #     "Cleaned caption:"
# # )

# # CLEAN_PROMPT_TEMPLATE = (
# #     "You are a helpful assistant that edits captions for ReID image data. "
# #     "Only return the cleaned caption describing the most prominent person. "
# #     "Remove background or unrelated information. Do not explain your answer.\n"
# #     "Original caption: \"{}\"\n"
# #     "Cleaned caption:"
# # )

# CLEAN_PROMPT_TEMPLATE = (
#     "Your task is to clean the following image caption so it only describes the main person, their appearance and clothing. "
#     "Remove any mention of the background or irrelevant elements. Be precise and brief. Do not include any explanations or context.\n\n"
#     "Original caption: \"{}\"\n"
#     "Cleaned caption:"
# )

# # === Load JoyCaption model ===
# # MODEL_NAME = "/mnt/database/hammad/hf_models/llava_joycaption_model/"
# # processor = AutoProcessor.from_pretrained(MODEL_NAME)
# # llava_model = LlavaForConditionalGeneration.from_pretrained(
# #     MODEL_NAME, torch_dtype=torch.bfloat16, device_map=0
# # )
# # llava_model.eval()

# # === Load small LLaMA3 or similar model for caption cleaning ===
# CLEAN_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change if you use another
# clean_tokenizer = AutoTokenizer.from_pretrained(CLEAN_MODEL_NAME)
# clean_model = AutoModelForCausalLM.from_pretrained(CLEAN_MODEL_NAME, device_map=0, torch_dtype=torch.float16)
# clean_model.eval()

# # === Image files ===
# image_list = [
#     f for f in os.listdir(IMAGE_DIR)
#     if f.lower().endswith((".jpg", ".jpeg", ".png"))
# ]

# # === Start processing ===
# with open(SAVE_PATH, "w", encoding="utf-8") as f_out:
#     # for i, img_name in enumerate(image_list[1]):
#     for i in range(0,1):
#         img_name = image_list[0]
#         img_path = os.path.join(IMAGE_DIR, img_name)
#         try:
#             image = Image.open(img_path).convert("RGB")
#         except UnidentifiedImageError:
#             print(f"[{i}] Skipped non-image file: {img_name}")
#             continue

#         # JoyCaption conversation setup
#         convo = [
#             {"role": "system", "content": "You are a helpful image captioner."},
#             {"role": "user", "content": GEN_PROMPT},
#         ]
#         # convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
#         # inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to("cuda")
#         # inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

#         # # JoyCaption generation
#         # with torch.no_grad():
#         #     start_time = time.time()
#         #     gen_ids = llava_model.generate(
#         #         **inputs,
#         #         max_new_tokens=300,
#         #         do_sample=True,
#         #         suppress_tokens=None,
#         #         use_cache=True,
#         #         temperature=0.6,
#         #         top_k=None,
#         #         top_p=0.9,
#         #     )[0]
            
            
#         #     gen_ids = gen_ids[inputs['input_ids'].shape[1]:]
#         #     raw_caption = processor.tokenizer.decode(
#         #         gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         #     ).strip()
#         #     caption_time = time.time() - start_time

#         raw_caption = "A man with a beard and glasses standing in front of a white building."

#         # Cleaning with LLaMA3-lite
#         clean_prompt = CLEAN_PROMPT_TEMPLATE.format(raw_caption)
#         clean_inputs = clean_tokenizer(clean_prompt, return_tensors="pt").to("cuda")

#         with torch.no_grad():
#             clean_output_ids = clean_model.generate(
#                 **clean_inputs,
#                 max_new_tokens=300,
#                 do_sample=False
#             )
#         clean_caption = clean_tokenizer.decode(
#             clean_output_ids[0][clean_inputs["input_ids"].shape[1]:],
#             skip_special_tokens=True
#         ).strip()

#         # print(f"[{i}] {img_name} | Caption: {clean_caption} | Time: {caption_time:.2f}s")
#         f_out.write(f"{img_name}\t{clean_caption}\n")


import os
import time
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import torch

# -------- CONFIG -------- #
JOY_MODEL_PATH = "/mnt/database/hammad/hf_models/llava_joycaption_model/"
# LLM_MODEL_NAME = "/mnt/data1/Hammad/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
# LLM_MODEL_NAME = "/mnt/database/hammad/hf_models/TinyLlama-1.1B"  # Change if you use another
# LLM_MODEL_NAME = "/mnt/database/hammad/hf_models/Llama-3.2-1B"
LLM_MODEL_NAME = "/mnt/database/hammad/hf_models/Llama-3.2-3B-Instruct"

IMAGE_DIR = "./"  # Folder with your test images
OUTPUT_FILE = "cleaned_captions.txt"
DEVICE = "cuda"

PROMPT = (
    "Write a short descriptive caption for this image, include features like what the person in the object has wear on upper body "
    "as t-shirt, shirt, coat, and lower body as trouser, pant, and what he wears, like joggers, slippers, and what he has on his head, "
    "a cap etc, and her hair colour if visible, age, ethnicity."
)

# CLEAN_PROMPT_TEMPLATE = (
#     "Your task is to clean the following image caption so it only describes the main person, their appearance and clothing. "
#     "Remove any mention of the background or irrelevant elements. Be precise and brief. Do not include any explanations or context.\n\n"
#     "Original caption: \"{}\"\n"
#     "Cleaned caption:"
# )

# CLEAN_PROMPT_TEMPLATE = """
# Your task is to rewrite the following image caption to make it suitable for person re-identification (ReID). 
# Only describe the person's visible features such as clothing, accessories, hair, shoes, age, and body type. 
# Completely remove any mention of background, location, or surrounding objects. 
# Do not add anything that isn't visible in the original caption.

# Original Caption: "{}"

# ReID-focused Caption:
# """

# CLEAN_PROMPT_TEMPLATE = """
# You are given a raw image caption generated by an image captioning model. Your task is to rewrite the caption to be suitable for a person re-identification (ReID) system.

# Focus ONLY on the visual appearance of the **most prominent person** in the image. Include clear and concise information such as:

# - Upper body clothing (e.g. jacket, t-shirt, hoodie)
# - Lower body clothing (e.g. jeans, trousers)
# - Footwear (e.g. sneakers, boots)
# - Hair style and color (if visible)
# - Accessories (e.g. hat, bag, glasses)
# - Gender, age group (child, teenager, adult, elderly), and ethnicity **only if visually obvious**

# 🚫 Do NOT mention:
# - Background, scenery, lighting, location, weather
# - Emotions or abstract concepts
# - Objects not worn or carried by the person

# Note:
# - You are a helpful assistant that edits captions for ReID image data. 
# - Only return the cleaned caption describing the most prominent person.
# - Remove background or unrelated information. Do not explain your answer.
# - Don't include any explanations or context or examples or additional text ot note or original caption.
# - just return the cleaned caption.

# Here is the raw caption:
# "{raw_caption}"

# ReID caption:"""

# CLEAN_PROMPT_TEMPLATE = '''
# Only describe the most prominent person's appearance for person re-identification. 
# Include only:
# - Clothing (top, bottom, footwear)
# - Hair (style, color)
# - Accessories (hat, bag, etc.)
# - Gender, age group, and ethnicity if visually obvious.

# Do NOT mention background, scenery, objects, or anything else.

# Caption:
# "{raw_caption}"

# ReID caption:'''

CLEAN_PROMPT_TEMPLATE = """
You are given a raw image caption generated by an image captioning model. Your task is to rewrite the caption to be suitable for a person re-identification (ReID) system.

Describe only the most prominent person in the image, using natural language.

✅ Your description should include:
- Clothing (upper, lower, footwear)
- Hair style and color (if visible)
- Accessories (e.g. hat, bag, glasses)
- Gender, age group, and ethnicity — only if visually obvious

🚫 Do NOT include:
- Background, scenery, location, weather
- Emotions or objects not worn or carried
- backgground objects like cycle, car, etc

Format:
Return a single, natural-language sentence that clearly and fluently describes the person’s appearance. No bullet points, no labels, no lists.

Here is the raw caption:
"{raw_caption}"

ReID caption:"""

# -------- Load Models -------- #

print("Loading LLaMA3-lite model for cleaning...")
clean_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
clean_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE).eval()

print("Loading JoyCaption model...")
processor = AutoProcessor.from_pretrained(JOY_MODEL_PATH)
joy_model = LlavaForConditionalGeneration.from_pretrained(JOY_MODEL_PATH, torch_dtype=torch.bfloat16, device_map=0).eval()


# -------- Process Images -------- #
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

with open(OUTPUT_FILE, "w") as f_out:
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{i}] Skipped {img_name}: Not a valid image - {e}")
            continue

        # Prepare JoyCaption input
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": PROMPT},
        ]
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        # Encode inputs
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(DEVICE)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # Generate raw caption
        with torch.no_grad():
            output_ids = joy_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )[0]
            output_ids = output_ids[inputs["input_ids"].shape[1]:]
            raw_caption = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Clean the caption
        # clean_prompt = CLEAN_PROMPT_TEMPLATE.format(raw_caption)
        clean_prompt = CLEAN_PROMPT_TEMPLATE.format(raw_caption=raw_caption)
        clean_inputs = clean_tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)

        # clean_inputs = clean_tokenizer(clean_prompt, return_tensors="pt")
        # clean_inputs = {k: v.to(DEVICE) for k, v in clean_inputs.items()}

        with torch.no_grad():
            clean_output_ids = clean_model.generate(
                **clean_inputs,
                max_new_tokens=100,
                do_sample=False,
                eos_token_id=clean_tokenizer.eos_token_id
            )

        # clean_caption = clean_tokenizer.decode(
        #     clean_output_ids[0][clean_inputs["input_ids"].shape[1]:],
        #     skip_special_tokens=True
        # ).strip()
        decoded = clean_tokenizer.decode(clean_output_ids[0], skip_special_tokens=True)

        # Remove everything before and including "ReID caption:"
        if "ReID caption:" in decoded:
            clean_caption = decoded.split("ReID caption:")[-1].strip()
        else:
            clean_caption = decoded.strip()

        clean_caption = clean_caption.split("\n")[0].strip().strip('"')  # also strip quotes if present

        # Write to file
        # f_out.write(f"{img_name}\t{raw_caption}\t{clean_caption}\n")
        f_out.write(f"{img_name}\t{clean_caption}\n")
        # print(f"[{i}] {img_name} → {raw_caption} → {clean_caption}")


# # CUDA_VISIBLE_DEVICES=2 python captioning_pipeline.py