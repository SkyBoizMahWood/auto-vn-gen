import copy
from json.decoder import JSONDecodeError

from loguru import logger

from src.llms.llm import LLM
from src.models.generation_context import GenerationContext
from src.prompts.utility_prompts import get_fix_invalid_json_prompt
from src.types.openai import ConversationHistory
from src.utils.general import parse_json_string
from src.utils.openai_ai import append_openai_message


class TunedLlamaModel(LLM):
    def __init__(self, model_name: str, max_tokens: int = 32768):
        super().__init__(max_tokens)
        self.model_name = model_name
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "Soraki5th/auto-vn-gen-llama-8b-v2", # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = 32768,
            dtype = None,
            load_in_4bit = True,
        )
        
        self.model = model
        self.tokenizer = tokenizer

        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference

    def count_token(self, message: str) -> int:
        return self.tokenizer(message, return_tensors="pt")["input_ids"].shape[1]

    @staticmethod
    def get_history_message(messages: list) -> str:
        history = ""
        for message in messages:
            history += f"{message['content']} "
        return history

    def get_last_assistant_response(self, conversation):
        # Split the conversation into message blocks
        messages = conversation[0].split('<|eot_id|>')
        
        assistant_responses = []
        
        for msg in messages:
            if '<|start_header_id|>assistant' in msg:
                # Extract the content after the header
                content = msg.split('<|end_header_id|>')[-1].strip()
                assistant_responses.append(content)
        
        # Return the last assistant response if any exist
        return assistant_responses[-1] if assistant_responses else None
    
    def convert_to_normal_messages(self, messages: ConversationHistory) -> list:
        converted_messages = []
        for message in messages:
            converted_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        return converted_messages
            
    
    def generate_content(self, ctx: GenerationContext, messages: ConversationHistory) -> tuple[str, dict]:
        logger.debug(f"Starting chat completion with model: {self.model_name}")

        copied_messages = copy.deepcopy(messages)

        copied_messages = self.rolling_history(copied_messages)

        copied_messages = self.convert_to_normal_messages(copied_messages)

        try:
            inputs = self.tokenizer.apply_chat_template(
                copied_messages,
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")

            outputs = self.model.generate(input_ids = inputs, max_new_tokens = 8192,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

            response = self.get_last_assistant_response(self.tokenizer.batch_decode(outputs))
            prompt_tokens = self.count_token(self.get_history_message(copied_messages))
            completion_tokens = self.count_token(response)

            parsed_response = parse_json_string(response)

            ctx.append_response_to_file(self.model_name, response, prompt_tokens, completion_tokens)
            ctx.append_history_to_file(copied_messages)

            return response, parsed_response
        except (ValueError, JSONDecodeError) as e:
            logger.warning(f"Tuned Llama Model response could not be decoded as JSON: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise e

    def fix_invalid_json_generation(self, ctx: GenerationContext, old_response: str, error_msg: str) -> tuple[
        str, dict]:
        fix_json_prompt = get_fix_invalid_json_prompt(old_response, error_msg)
        retry_history = append_openai_message("You are a helpful coding AI assistant.", "system")
        retry_history = append_openai_message(fix_json_prompt, "user", retry_history)
        logger.warning(f"Retrying with: {retry_history}")

        return self.generate_content(ctx, retry_history)

    def __str__(self):
        return f"TunedModel(model_name={self.model_name}, max_tokens={self.max_tokens})"
