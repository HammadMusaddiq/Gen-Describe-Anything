import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import time

# IMAGE_PATH = "RGBNT201_TI_000018_cam1_0_04.jpg"
# PROMPT = "Write a short descriptive caption for this image in a formal tone."
# MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

MODEL_NAME = "/mnt/database/hammad/hf_models/llava_joycaption_model/"
PROMPT = "Write a short descriptive caption for this image, include features like what the person in the object have wear on upper body as t-shirt, shirt, coat, and lower body as trouser, pant, and what he wears, like joggers, sleepers, and what he has on his head, a cap etc, and her hair colour if visible, age, ethnicity."
Re_PROMPT = "remove any background image information from the caption."

# Load JoyCaption
# bfloat16 is the native dtype of the LLM used in JoyCaption (Llama 3.1)
# device_map=0 loads the model into the first GPU
processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="bfloat16", device_map=0)
llava_model.eval()

image_list = [
    "RGBNT201_TI_000018_cam1_0_04.jpg",
    "lwir_cv2_307_77_0.jpg",
    "pre_164_41_2.jpg",
    "pre_307_77_0.jpg",
    "org_307_77_0.jpg",
    "Weixin Image_20250422144646.jpg"
]

for image_path in image_list:

    image = Image.open(image_path).convert("RGB")

    # Build the conversation
    convo = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": PROMPT,
        },
    ]

    # Format the conversation
    # WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
    # but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
    # if not careful, which can make the model perform poorly.
    convo_string = processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
    # assert isinstance(convo_string, str)

    # Process the inputs
    inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

    # Start timing
    start_time = time.time()

    with torch.no_grad():
        # Load image
        # image = Image.open(IMAGE_PATH)

        # Generate the captions
        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=0.6,
            top_k=None,
            top_p=0.9,
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.strip()
        # print(caption)
    
    
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Image: {image_path}")
    print(f"Caption: {caption}")
    print(f"Inference Time: {inference_time:.2f} seconds\n")


# CUDA_VISIBLE_DEVICES=2 python joy_caption_gen.py