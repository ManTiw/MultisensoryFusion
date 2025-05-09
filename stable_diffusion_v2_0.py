import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import base64

# Configuration class
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    auth_token = "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"

# Load the model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token=CFG.auth_token
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_images(prompt):
    """
    Generate images using the Stable Diffusion 2.0 model for the given prompt.

    Args:
        prompt (str): The text prompt for generating images.

    Returns:
        list: A list of base64-encoded images.
    """
    try:
        # Generate images
        result = image_gen_model(
            prompt,
            num_inference_steps=CFG.image_gen_steps,
            generator=CFG.generator,
            guidance_scale=CFG.image_gen_guidance_scale
        )
        images = result.images

        # Resize images and convert to base64 for rendering in HTML
        image_data = []
        for img in images:
            resized_image = img.resize(CFG.image_gen_size)
            buffered = io.BytesIO()
            resized_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data.append(img_str)

        return image_data
    except Exception as e:
        print(f"Error in generating images: {e}")
        raise
