import copy
import os
from json import JSONDecodeError
from time import sleep

from google import genai
from google.genai import types
from google.genai.types import Content
from google.genai.errors import APIError
from google.genai.types import HarmCategory, HarmBlockThreshold
from loguru import logger

from src.llms.llm import LLM
from src.models.generation_context import GenerationContext
from src.prompts.utility_prompts import get_fix_invalid_json_prompt
from src.utils.general import parse_json_string
from ..types.openai import ConversationHistory
from ..utils.google_ai import map_openai_history_to_google_history, map_google_history_to_openai_history
from ..utils.openai_ai import append_openai_message


class GoogleModel(LLM):
    def __init__(self, model_name: str, max_tokens: int = 32768):
        super().__init__(max_tokens)
        api_keys_str = os.environ.get("GOOGLE_AI_API_KEY", "")
        self.api_keys = api_keys_str.split(",") if api_keys_str else []
        if not self.api_keys:
            raise ValueError("GOOGLE_AI_API_KEY environment variable not set or empty")
        logger.debug(f"Google API keys: {self.api_keys}")
        self.current_key_index = 0
        self.model_name = model_name
        self.client = genai.Client(
            api_key=self.api_keys[self.current_key_index], http_options=types.HttpOptions(api_version='v1alpha'))

    def count_token(self, message: str) -> int:
        count = self.client.models.count_tokens(model=self.model_name, contents=message).total_tokens
        return count

    @staticmethod
    def get_history_message(messages: list[Content]) -> str:
        history = ""
        for message in messages:
            history += f"{message.parts[0].text} "
        return history

    def generate_content(self, ctx: GenerationContext, messages: ConversationHistory) -> tuple[str, dict]:
        logger.debug(f"Starting chat completion with model: {self.model_name}")

        copied_messages = copy.deepcopy(messages)
        copied_messages = self.rolling_history(copied_messages)
        last_message = copied_messages.pop()
        if last_message.get("role") == "system" or last_message.get("role") == "assistant":
            raise ValueError(f"Last message role is not user: {last_message.get('role')}")
        current_message = last_message.get("content")

        copied_messages = map_openai_history_to_google_history(copied_messages)
        chat = self.client.chats.create(model=self.model_name, history=copied_messages)

        try:
            chat_completion = chat.send_message(message=current_message,
                                                config=types.GenerateContentConfig(
                                                    temperature=1.5,
                                                    top_k=1,
                                                    top_p=0.9,
                                                    safety_settings=[
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                        types.SafetySetting(
                                                            category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                                                            threshold=HarmBlockThreshold.OFF
                                                        ),
                                                    ]
                                                ))

            response = chat_completion.text.strip()

            copied_messages = copied_messages + map_openai_history_to_google_history([last_message])
            prompt_tokens = self.count_token(self.get_history_message(copied_messages))
            response_tokens = self.count_token(response)
            
            parsed_response = parse_json_string(response)

            ctx.append_response_to_file(self.model_name, response, prompt_tokens, response_tokens)
            ctx.append_history_to_file(map_google_history_to_openai_history(copied_messages))

            return response, parsed_response
        except (ValueError, JSONDecodeError) as e:
            logger.warning(f"Gemini Model: {self.model_name} response could not be decoded as JSON: {str(e)}")
            raise e
        except (APIError) as e:
            logger.warning(f"Gemini Model: {self.model_name} API error: {e}")
            # Switch to next API key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.client = genai.Client(
                api_key=self.api_keys[self.current_key_index], http_options=types.HttpOptions(api_version='v1alpha'))
            logger.warning(f"Switched to API key index {self.current_key_index}")
            sleep(3)
            return self.generate_content(ctx, messages)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise e

    def fix_invalid_json_generation(self, ctx: GenerationContext, old_response: str, error_msg: str) -> tuple[
        str, dict]:
        fix_json_prompt = get_fix_invalid_json_prompt(old_response, error_msg)
        retry_history = append_openai_message(fix_json_prompt, "user")
        logger.warning(f"Retrying with: {retry_history}")

        return self.generate_content(ctx, retry_history)

    def __str__(self):
        return f"Gemini(model_name={self.model_name}, max_tokens={self.max_tokens})"
