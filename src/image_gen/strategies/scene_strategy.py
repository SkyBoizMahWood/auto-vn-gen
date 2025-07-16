from src.image_gen.strategies.base_strategy import ImageGenerationStrategy
from src.types.image_gen import ImageShape
import requests
from loguru import logger
from src.prompts.image_prompts import get_scene_negative_image_prompt


class SceneGenerationStrategy(ImageGenerationStrategy):
    def generate(self, model: "LocalStableDiffusionModel", prompt: str, shape: ImageShape) -> str:
        model.set_checkpoint("crystalClearXL_ccxl.safetensors")
        size = {
            "landscape": {"width": 1344, "height": 768}
        }
        response = requests.post(model.text_2_img_api_url, json={
            "prompt": f"<lora:niji_vn_bg:1>, niji_vn_bg, no humans,\n{prompt}",
            "negative_prompt": get_scene_negative_image_prompt(),
            "width": size[shape]["width"],
            "height": size[shape]["height"],
            "steps": 20,
            "sampler_name": "DPM++ 2M",
            "scheduler": "Karras",
            "cfg_scale": 8,
            "denoising_strength": 0.85,
            "seed": -1,
        }, timeout=100000)
        if response.status_code == 200:
            logger.debug("Scene image generated successfully.")
            return response.json().get("images")[0]
        else:
            logger.error(f"Failed to generate scene image: {response.text}")
            return None