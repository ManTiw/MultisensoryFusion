import torch
from diffusers import DiffusionPipeline
from PIL import Image
import io
import base64

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.to(device)

def generate_images(prompt):
    """
    Generate images using the Stable Diffusion v1.5 model for the given prompt.

    Args:
        prompt (str): The text prompt for generating images.

    Returns:
        list: A list of base64-encoded images.
    """
    try:
        # Generate images
        result = model(prompt, num_inference_steps=10, height=512, width=512, num_images_per_prompt=5)
        images = result.images

        # Convert images to base64 for rendering in HTML
        image_data = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data.append(img_str)

        return image_data
    except Exception as e:
        print(f"Error in generating images: {e}")
        raise
