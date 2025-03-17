from src.image_gen.strategies.base_strategy import ImageGenerationStrategy
from src.types.image_gen import ImageShape
import requests
from loguru import logger


class CharacterGenerationStrategy(ImageGenerationStrategy):
    def generate(self, model: "LocalStableDiffusionModel", prompt: str, shape: ImageShape) -> str:
        model.set_checkpoint("novaAnimeXL_ilV50.safetensors")
        size = {
            "portrait": {"width": 768, "height": 1344},
            "square": {"width": 1024, "height": 1024}
        }
        response = requests.post(model.text_2_img_api_url, json={
            "prompt": f"Character: {prompt}",
            "negative_prompt": "blurry, low quality, bad anatomy",
            "width": size[shape]["width"],
            "height": size[shape]["height"],
            "steps": 40,
            "sampler_name": "Euler a",
            "cfg_scale": 7,
            "denoising_strength": 0.7
        })
        if response.status_code == 200:
            logger.debug("Character image generated successfully.")
            return response.json().get("images")[0]
        else:
            logger.error(f"Failed to generate character image: {response.text}")
            return None