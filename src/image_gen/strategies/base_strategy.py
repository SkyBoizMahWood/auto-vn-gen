from abc import ABC, abstractmethod
from src.types.image_gen import ImageShape

class ImageGenerationStrategy(ABC):
    @abstractmethod
    def generate(self, model: "LocalStableDiffusionModel", prompt: str, shape: ImageShape) -> str:
        """
        Generate an image using the given model, prompt, and shape.
        :param model: The LocalStableDiffusionModel instance.
        :param prompt: The text prompt for image generation.
        :param shape: The shape of the image (portrait, landscape, square).
        :return: The generated image as a base64 string.
        """
        pass