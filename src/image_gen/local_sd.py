import os

import requests
from loguru import logger

from src.image_gen.image_gen_model import ImageGenModel
from src.prompts.image_prompts import get_negative_image_prompt
from src.types.image_gen import ImageShape


class LocalStableDiffusionModel(ImageGenModel):
    def __init__(self):
        self.api_url = os.getenv("LOCAL_SD_BASE_URL")
        self.checkpoint_character = "novaAnimeXL_ilV50.safetensors"
        self.checkpoint_background = "crystal_clear_XL.safetensors"
        
    def set_checkpoint(self, checkpoint: str):
        """Change checkpoint before generating image"""
        logger.debug(f"Switching to checkpoint: {checkpoint}")
        response = requests.post(f"{self.api_url}/sdapi/v1/options", json={
            "sd_model_checkpoint": checkpoint
        })
        if response.status_code == 200:
            logger.debug("Checkpoint switched successfully.")
        else:
            logger.error("Failed to switch checkpoint.")

    def generate_image_from_text_prompt(self, prompt: str, shape: ImageShape = "square"):
        logger.debug(f"Generating image from prompt: {prompt}")

        size = {
            "portrait": {
                'width': 768,
                'height': 1344
            },
            "landscape": {
                'width': 1344,
                'height': 768
            },
            "square": {
                'width': 1024,
                'height': 1024
            }
        }
        
        if shape in ["portrait", "square"]:
            self.set_checkpoint(self.checkpoint_character)
        else:
            self.set_checkpoint(self.checkpoint_background)

        response = requests.post(f"{self.api_url}/sdapi/v1/txt2img", json={
            "prompt": prompt,
            "negative_prompt": get_negative_image_prompt(),
            "width": size[shape]["width"],
            "height": size[shape]["height"],
            "steps": 50,
            "sampler_name": "Euler a",
            "denoising_strength": 0.8,
            "cfg_scale": 7,
        })

        return response.json().get("images")[0]

    def __str__(self):
        return "LocalStableDiffusionModel"
