from loguru import logger
from typing_extensions import Optional

from src.image_gen.dall_e_three import DallEThree
from src.image_gen.image_gen_model import ImageGenModel
from src.image_gen.local_sd import LocalStableDiffusionModel
from src.image_gen.stable_cascade import StableCascade
from src.llms.anthropic_model import AnthropicModel
from src.llms.google_model import GoogleModel
from src.llms.llm import LLM
from src.llms.openai_model import OpenAIModel
from src.llms.tuned_llama_model import TunedLlamaModel
from src.llms.tuned_llama_model_vllm import TunedLlamaModelVLLM

MAX_TOKENS = {
    'gpt-3.5-turbo-0125': 16385,
    'gpt-4-turbo-preview': 128000,
    'gemini-1.0-pro': 32768,
    'gemini-1.5-flash': 1048576,
    'gemini-2.0-flash-exp': 1048576,
    'gemini-2.0-flash': 1048576,
    'gemini-2.0-flash-001': 1048576,
    'gemini-2.0-flash-thinking-exp': 1048576,
    'gemini-2.5-pro-preview-05-06': 1048576,
    'gemini-2.5-flash-preview-04-17': 1048576,
    'claude-3-opus-20240229': 200000,
    'claude-3-sonnet-20240229': 200000,
    'claude-2.1': 200000,
    'tuned-llama-8b': 32768,
    'tuned-llama-8b-vllm': 32768,
}


def get_generation_model(model_name: str, seed: Optional[int]) -> LLM:
    if model_name in ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"]:
        max_tokens = MAX_TOKENS[model_name]
        return OpenAIModel(model_name, max_tokens, seed)
    elif model_name in ["gemini-1.0-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-2.0-flash-thinking-exp", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-04-17"]:
        if seed is not None:
            logger.warning(f"Seed is set for model {model_name}, but it will be ignored.")
        max_tokens = MAX_TOKENS[model_name]
        return GoogleModel(model_name, max_tokens)
    elif model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1"]:
        if seed is not None:
            logger.warning(f"Seed is set for model {model_name}, but it will be ignored.")
        max_tokens = MAX_TOKENS[model_name]
        return AnthropicModel(model_name, max_tokens)
    elif model_name in ["tuned-llama-8b"]:
        max_tokens = MAX_TOKENS[model_name]
        return TunedLlamaModel(model_name, max_tokens)
    elif model_name in ["tuned-llama-8b-vllm"]:
        max_tokens = MAX_TOKENS[model_name]
        return TunedLlamaModelVLLM(model_name, max_tokens)
    else:
        raise ValueError(f"Unknown generation model: {model_name}")


def get_image_generation_model(model_name: str) -> ImageGenModel:
    if model_name in ["dall-e-3"]:
        return DallEThree()
    elif model_name in ['local-sd']:
        return LocalStableDiffusionModel()
    elif model_name in ["stable-cascade"]:
        return StableCascade()
    else:
        raise ValueError(f"Unknown image generation model: {model_name}")
