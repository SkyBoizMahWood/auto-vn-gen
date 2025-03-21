import os

import requests
from loguru import logger
from src.image_gen.strategies.base_strategy import ImageGenerationStrategy
from src.image_gen.strategies.character_strategy import CharacterGenerationStrategy
from src.image_gen.strategies.scene_strategy import SceneGenerationStrategy
from src.image_gen.image_gen_model import ImageGenModel
from src.types.image_gen import ImageShape

class StrategyContext:
    def __init__(self, strategy: ImageGenerationStrategy = None):
        self.strategy = strategy

    def set_strategy(self, strategy: ImageGenerationStrategy):
        self.strategy = strategy

    def execute_strategy(self, model, prompt: str, shape: ImageShape):
        if not self.strategy:
            raise ValueError("No strategy set for image generation.")
        return self.strategy.generate(model, prompt, shape)

class LocalStableDiffusionModel(ImageGenModel):
    def __init__(self):
        self.text_2_img_api_url = f"{os.getenv('LOCAL_SD_BASE_URL')}/sdapi/v1/txt2img"
        self.switch_checkpoint_api_url = f"{os.getenv('LOCAL_SD_BASE_URL')}/sdapi/v1/options"
        self.strategy_context = StrategyContext()
        
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
            self.strategy_context.set_strategy(CharacterGenerationStrategy())
        elif shape == "landscape":
            self.strategy_context.set_strategy(SceneGenerationStrategy())
        else:
            raise ValueError(f"Unsupported shape: {shape}")
        
        return self.strategy_context.execute_strategy(self, prompt, shape)
