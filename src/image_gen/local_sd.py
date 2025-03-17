import os

import requests
from loguru import logger
from src.image_gen.strategies.base_strategy import ImageGenerationStrategy
from src.image_gen.strategies.character_strategy import CharacterGenerationStrategy
from src.image_gen.strategies.scene_strategy import SceneGenerationStrategy
from src.image_gen.image_gen_model import ImageGenModel
from src.types.image_gen import ImageShape

class LocalStableDiffusionModel(ImageGenModel):
    def __init__(self):
        self.text_2_img_api_url = f"{os.getenv('LOCAL_SD_BASE_URL')}/sdapi/v1/txt2img"
        self.switch_checkpoint_api_url = f"{os.getenv('LOCAL_SD_BASE_URL')}/sdapi/v1/options"
        self.strategy = None
        
    def set_checkpoint(self, checkpoint: str):
        logger.debug(f"Switching to checkpoint: {checkpoint}")
        response = requests.post(self.switch_checkpoint_api_url, json={
            "sd_model_checkpoint": checkpoint
        })
        if response.status_code == 200:
            logger.debug("Checkpoint switched successfully.")
        else:
            logger.error("Failed to switch checkpoint." + response.text)

    def generate_image_from_text_prompt(self, prompt: str, shape: ImageShape):
        if shape in ["portrait", "square"]:
            self.strategy = CharacterGenerationStrategy()
        elif shape == "landscape":
            self.strategy = SceneGenerationStrategy()
        else:
            raise ValueError(f"Unsupported shape: {shape}")
        
        return self.strategy.generate(self, prompt, shape)
