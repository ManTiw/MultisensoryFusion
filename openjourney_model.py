import os
import torch
import io
import base64
from PIL import Image
from diffusers import StableDiffusionPipeline

# Load Hugging Face token from environment
auth_token = os.getenv("Your HF Auth Token")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    "prompthero/openjourney",
    use_auth_token=auth_token
)
pipe = pipe.to(device)

def generate_images(prompt):
    result = pipe(prompt, num_inference_steps=30)
    images = result.images

    image_data = []
    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data.append(img_str)

    return image_data
