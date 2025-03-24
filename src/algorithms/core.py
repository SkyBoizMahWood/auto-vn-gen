import json
import os

from loguru import logger

from src.models.enums.branching_type import BranchingType
from src.models.generation_context import GenerationContext
from src.models.story.story_choice import StoryChoice
from src.models.story_chunk import StoryChunk
from src.models.story_data import StoryData
from src.prompts.image_prompts import (get_character_image_prompt,
                                       get_scene_image_prompt)
from src.prompts.story_prompts import (
    get_plot_prompt, get_story_based_on_selected_choice_prompt,
    get_story_until_chapter_end_prompt,
    get_story_until_choices_opportunity_prompt,
    get_story_until_game_end_prompt)
from src.utils.general import get_base64_from_image, get_image_from_base64
from src.utils.openai_ai import append_openai_message


def initialize_generation(ctx: GenerationContext):
    logger.debug("Start story plot generation")

    game_story_prompt = get_plot_prompt(ctx.config)
    history = append_openai_message(game_story_prompt)

    with open(ctx.output_path / "histories.json", "w") as file:
        file.write(json.dumps({"histories": [history]}, indent=2))

    if not ctx.config.existing_plot:
        max_retry_attempts = 10
        has_generation_success, current_attempt = False, 0
        story_data_raw, story_data_obj = None, None
        while not has_generation_success and current_attempt < max_retry_attempts:
            try:
                story_data_raw, story_data_obj = ctx.generation_model.generate_content(ctx, history)
                has_generation_success = True
            except Exception as e:
                current_attempt += 1
                logger.warning(f"Exception occurred while chat completion: {e}")
                logger.warning(f"Retry {current_attempt}/{max_retry_attempts}")
    
        if not has_generation_success or story_data_raw is None or story_data_obj is None:
            logger.error(f"Failed to generate story data.")
            logger.error("Exiting...")
            exit(1)
    else:
        with open(ctx.config.existing_plot, "r") as file:
            content = json.load(file)
            story_data_raw = content["raw"]
            story_data_obj = content["parsed"]
    try:
        story_data_obj["id"] = ctx.story_id
        story_data_obj["generated_by"] = os.getenv("GENERATION_MODEL")
        story_data_obj["approach"] = ctx.approach
        story_data = StoryData.from_json(story_data_obj)
    except Exception as e:
        logger.warning("Can not load json parsed from response. Retrying generate story data...")
        return initialize_generation(ctx)

    if ctx.config.enable_image_generation and not ctx.config.existing_plot:
        logger.debug("Start character image generation")
        for character in story_data.main_characters:
            logger.debug(f"Generating image for character: {character}")

            prompt = get_character_image_prompt(character)
            image_b64 = ctx.image_gen_model.generate_image_from_text_prompt(prompt, shape="square", )

            character.original_image = image_b64

            # image = get_image_from_base64(image_b64)
            # removed_bg_image = ctx.background_remover_model.remove_background(image)
            # character.image = get_base64_from_image(removed_bg_image)
            character.image = image_b64
            logger.debug(f"Generated image for character: {character}")

        logger.debug("Start scene image generation")
        for scene in story_data.main_scenes:
            logger.debug(f"Generating image for scene: {scene}")

            prompt = "<lora:PE_AnimeBG:1>" + get_scene_image_prompt(scene)
            image_b64 = ctx.image_gen_model.generate_image_from_text_prompt(prompt, shape="landscape")

            scene.image = image_b64
            logger.debug(f"Generated image for scene: {scene}")
    else:
        logger.debug("Image generation is disabled")

    ctx.db_connector.write(story_data)

    initial_history = append_openai_message(story_data_raw, role="assistant", history=history)
    logger.debug("End story plot generation")

    with open(ctx.output_path / "plot.json", "w") as file:
        file.write(
            json.dumps({"raw": story_data_raw, "parsed": story_data.to_json()}, indent=2)
        )

    return initial_history, story_data


def get_prompts_by_branching_type(choice: StoryChoice, ctx: GenerationContext, current_chapter: int, current_num_choices: int,
                                  parent_chunk: StoryChunk, state: BranchingType, story_data: StoryData, used_choice_opportunity: int) -> str:
    if state is BranchingType.BRANCHING:
        if not choice:  # Start of chapter
            prompt = get_story_until_choices_opportunity_prompt(ctx.config, story_data, current_num_choices,
                                                                used_choice_opportunity, current_chapter)
        else:  # In the middle of chapter
            prompt = get_story_based_on_selected_choice_prompt(ctx.config, story_data, choice, current_num_choices,
                                                               used_choice_opportunity, current_chapter)
    elif state is BranchingType.CHAPTER_END:
        prompt = get_story_until_chapter_end_prompt(ctx.config, story_data, parent_chunk)
    elif state is BranchingType.GAME_END:
        prompt = get_story_until_game_end_prompt(ctx.config, story_data, parent_chunk)
    else:
        logger.error(f"Invalid state: {state}")
        exit(1)
    return prompt
