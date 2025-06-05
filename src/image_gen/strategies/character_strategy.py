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
            "prompt": f"masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, Character: {prompt}, BREAK, depth of field, volumetric lighting",
            "negative_prompt": "modern, recent, old, oldest, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, long body, lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, very displeasing, (worst quality, bad quality:1.2), bad anatomy, sketch, jpeg artifacts, signature, watermark, username, signature, simple background, conjoined,bad ai-generated",
            "width": size[shape]["width"],
            "height": size[shape]["height"],
            "steps": 25,
            "sampler_name": "Euler a",
            "cfg_scale": 5,
            "denoising_strength": 0.7
        }, timeout=100000)
        if response.status_code == 200:
            logger.debug("Character image generated successfully.")
            return response.json().get("images")[0]
        else:
            logger.error(f"Failed to generate character image: {response.text}")
            return None