# auto-vn-gen/api.py
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger
import os
import json
from datetime import datetime

# Import necessary functions from your existing codebase
from src.batch_generation.core import (
    run_batch_generation,
)
from src.generation.core import run_generation_with
from src.models.enums.generation_approach import GenerationApproach
from src.models.generation_config import GenerationConfig
from src.utils.validators import validate_config, validate_existing_plot
from src.databases.neo4j import Neo4JConnector
from src.models.story.character_data import CharacterData
from src.models.story.scene_data import SceneData
from src.prompts.image_prompts import get_character_image_prompt, get_scene_image_prompt
from src.utils.generative_models import get_image_generation_model

# --- Refactored Script Logic ---

# From scripts/calculate.py
LLM_PRICES = {
    'claude-3-opus-20240229': {
        "prompt": 15 / 10e5,
        "completion": 75 / 10e5
    },
    "claude-3-sonnet-20240229": {
        "prompt": 3 / 10e5,
        "completion": 15 / 10e5
    },
    "claude-2.1": {
        "prompt": 8 / 10e5,
        "completion": 24 / 10e5
    },
    'gemini-1.0-pro': {
        "prompt": 0 / 10e5,
        "completion": 0 / 10e5
    },
    'gemini-1.5-flash': {
        "prompt": 0.075 / 10e5,
        "completion": 0.30 / 10e5
    },
    'gemini-2.0-flash-exp': {
        "prompt": 0.10 / 10e5,
        "completion": 0.40 / 10e5
    },
    'gemini-2.0-flash': {
        "prompt": 0.10 / 10e5,
        "completion": 0.40 / 10e5
    },
    'gemini-2.0-flash-001': {
        "prompt": 0.10 / 10e5,
        "completion": 0.40 / 10e5
    },
    'gemini-2.0-flash-thinking-exp': {
        "prompt": 0.10 / 10e5,
        "completion": 0.40 / 10e5
    },
    'gpt-3.5-turbo-0125': {
        "prompt": 0.5 / 10e5,
        "completion": 1.5 / 10e5
    },
    'gpt-4-0125-preview': {
        "prompt": 10 / 10e5,
        "completion": 30 / 10e5
    },
    'tuned-llama-8b-vllm': {
        "prompt": 0.03 / 10e5,
        "completion": 0.05 / 10e5
    }
}

def calculate_story_cost_logic(story_id: str):
    output_dir = Path("outputs") / story_id
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory for story_id {story_id} not found.")

    files = list(output_dir.glob("*.json"))
    model_name = None
    responses_path = None
    for file in files:
        if file.stem not in ["context", "histories", "plot"]:
            responses_path = file
            model_name = file.stem
            break
    
    if not responses_path or not model_name:
        raise FileNotFoundError(f"Response JSON file not found for story_id {story_id}.")

    plot_path = output_dir / "plot.json"
    context_path = output_dir / "context.json"

    if not plot_path.exists() or not context_path.exists():
        raise FileNotFoundError(f"plot.json or context.json not found for story_id {story_id}.")

    with open(responses_path, "r") as f:
        responses = json.load(f)
    with open(plot_path, "r") as f:
        plot_data = json.load(f)
    with open(context_path, "r") as f:
        context_data = json.load(f)

    prompt_tokens = responses.get("prompt_tokens", 0)
    completion_tokens = responses.get("completion_tokens", 0)

    if model_name not in LLM_PRICES:
        raise ValueError(f"LLM model '{model_name}' not found in price list.")

    prompt_token_price = LLM_PRICES[model_name]["prompt"]
    completion_token_price = LLM_PRICES[model_name]["completion"]

    prompt_cost = prompt_tokens * prompt_token_price
    completion_cost = completion_tokens * completion_token_price
    total_llm_cost = prompt_cost + completion_cost

    image_gen_model_name = context_data.get('image_generation_model', 'Unknown')
    price_per_character_image = 0
    price_per_scene_image = 0
    if image_gen_model_name == "DALL·E 3":
        price_per_character_image = 0.080
        price_per_scene_image = 0.120
    
    # Ensure plot_data["parsed"] exists and has expected keys
    parsed_plot = plot_data.get("parsed", {})
    num_characters = len(parsed_plot.get("main_characters", []))
    num_scenes = len(parsed_plot.get("main_scenes", []))

    character_images_cost = num_characters * price_per_character_image
    scene_images_cost = num_scenes * price_per_scene_image
    total_image_cost = character_images_cost + scene_images_cost

    total_cost = total_llm_cost + total_image_cost

    return {
        "story_id": story_id,
        "llm_model_name": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_cost": round(prompt_cost, 2),
        "completion_cost": round(completion_cost, 2),
        "total_llm_cost": round(total_llm_cost, 2),
        "image_generation_model": image_gen_model_name,
        "num_characters_generated": num_characters,
        "num_scenes_generated": num_scenes,
        "character_images_cost": round(character_images_cost, 2),
        "scene_images_cost": round(scene_images_cost, 2),
        "total_image_cost": round(total_image_cost, 2),
        "total_estimated_cost": round(total_cost, 2)
    }

def get_story_completion_time_logic(story_id: str):
    context_path = Path("outputs") / story_id / "context.json"
    if not context_path.exists():
        raise FileNotFoundError(f"context.json not found for story_id {story_id}.")

    with open(context_path, "r") as f:
        context = json.load(f)
    
    created_at_str = context.get("created_at")
    updated_at_str = context.get("updated_at")
    completed_at_str = context.get("completed_at")

    if not created_at_str or not updated_at_str:
        raise ValueError("Timestamps missing in context.json")

    created_at = datetime.fromisoformat(created_at_str)
    updated_at = datetime.fromisoformat(updated_at_str)
    
    time_to_completion_str = None
    time_to_last_update_str = None

    if completed_at_str:
        completed_at = datetime.fromisoformat(completed_at_str)
        time_to_completion_delta = completed_at - created_at
        h, rem = divmod(time_to_completion_delta.total_seconds(), 3600)
        m, s = divmod(rem, 60)
        time_to_completion_str = f"{int(h)}h {int(m)}m {int(s)}s"
    
    time_to_last_update_delta = updated_at - created_at
    h_update, rem_update = divmod(time_to_last_update_delta.total_seconds(), 3600)
    m_update, s_update = divmod(rem_update, 60)
    time_to_last_update_str = f"{int(h_update)}h {int(m_update)}m {int(s_update)}s"

    return {
        "story_id": story_id,
        "created_at": created_at.isoformat(),
        "updated_at": updated_at.isoformat(),
        "completed_at": completed_at.isoformat() if completed_at_str else None,
        "time_to_completion": time_to_completion_str,
        "time_to_last_update": time_to_last_update_str
    }

# From scripts/regenerate-images.py
def regenerate_story_images_logic(story_id: str, for_characters: bool, for_scenes: bool):
    img_gen_model_name = os.getenv('IMAGE_GENERATION_MODEL')
    if not img_gen_model_name:
        raise ValueError("IMAGE_GENERATION_MODEL environment variable not set.")
    img_gen = get_image_generation_model(img_gen_model_name)
    
    db = Neo4JConnector()
    
    def _regenerate(session):
        result_data = session.run("MATCH (n: StoryData {id: $story_id}) RETURN n LIMIT 1", story_id=story_id).data()
        if not result_data:
            raise FileNotFoundError(f"StoryData with id {story_id} not found in Neo4j.")
        story_node = result_data[0].get('n')

        if for_characters:
            main_characters_json = story_node.get('main_characters')
            if main_characters_json:
                main_characters = json.loads(main_characters_json)
                for character in main_characters:
                    character_data = CharacterData.from_json(character)
                    prompt = get_character_image_prompt(character_data)
                    image = img_gen.generate_image_from_text_prompt(prompt, shape="square")
                    character['original_image'] = image # Assuming image is base64 string
                    character['image'] = image # Placeholder if no background removal
                
                session.run("MATCH (n: StoryData {id: $id}) SET n.main_characters = $main_characters",
                            id=story_id, main_characters=json.dumps(main_characters))
                logger.info(f"Regenerated character images for story {story_id}")

        if for_scenes:
            main_scenes_json = story_node.get('main_scenes')
            if main_scenes_json:
                main_scenes = json.loads(main_scenes_json)
                for scene in main_scenes:
                    scene_data = SceneData.from_json(scene)
                    prompt = get_scene_image_prompt(scene_data)
                    image = img_gen.generate_image_from_text_prompt(prompt, shape="landscape")
                    scene['image'] = image # Assuming image is base64 string

                session.run("MATCH (n: StoryData {id: $id}) SET n.main_scenes = $main_scenes",
                            id=story_id, main_scenes=json.dumps(main_scenes))
                logger.info(f"Regenerated scene images for story {story_id}")

        # Update context file
        # Note: The original script uses result.get('n').get('approach') which might not always be present or correct.
        # We might need a more robust way to determine the output path or assume a standard one.
        approach_type = story_node.get('approach', 'unknown_approach') # Default if not found
        output_base = Path("outputs") / approach_type / story_id
        output_base.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        context_file = output_base / "context.json"
        context_data = {}
        if context_file.exists():
            with open(context_file, 'r') as file:
                context_data = json.load(file)
        
        context_data['image_generation_model'] = str(img_gen)
        context_data['last_image_regeneration'] = datetime.now().isoformat()
        with open(context_file, 'w') as file:
            json.dump(context_data, file, indent=2)
        
        return {"message": "Image regeneration process completed.", "characters_processed": for_characters, "scenes_processed": for_scenes}

    return db.with_session(_regenerate)

# From scripts/prune.py
def delete_story_data_logic(story_id: str):
    db = Neo4JConnector()
    def _prune(session):
        res_sd = session.run('MATCH (sd:StoryData {id: $story_id}) DETACH DELETE sd', story_id=story_id).consume()
        res_sc = session.run('MATCH (sc:StoryChunk {story_id: $story_id}) DETACH DELETE sc', story_id=story_id).consume()
        logger.info(f"Pruned StoryData nodes: {res_sd.counters.nodes_deleted}, StoryChunk nodes: {res_sc.counters.nodes_deleted} for story_id {story_id}")
        return {"nodes_deleted_story_data": res_sd.counters.nodes_deleted, "nodes_deleted_story_chunk": res_sc.counters.nodes_deleted}
    
    return db.with_session(_prune)

# --- FastAPI App Setup ---
app = FastAPI(
    title="Auto VN Generation API",
    description="API for generating visual novel stories and managing related data.",
    version="0.1.0"
)

# Load environment variables and set up logging
load_dotenv()
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True) # Ensure logs directory exists
logger.add(logs_dir / "api_{time}.log", rotation="1 day", retention="7 days")


# --- Pydantic Models for API ---
class StoryGenerationRequest(BaseModel):
    game_genre: Optional[str] = "visual novel"
    themes: Optional[List[str]] = None
    num_chapters: Optional[int] = 3
    num_endings: Optional[int] = 3
    num_main_characters: Optional[int] = 5
    num_main_scenes: Optional[int] = 5
    min_num_choices: Optional[int] = 2
    max_num_choices: Optional[int] = 3
    min_num_choices_opportunity: Optional[int] = 2
    max_num_choices_opportunity: Optional[int] = 3
    existing_plot: Optional[str] = None
    enable_image_generation: Optional[bool] = False
    seed: Optional[int] = None

class BatchGenerationRequest(StoryGenerationRequest):
    n_stories: Optional[int] = 10 # Default changed from 50 to 10 for sanity

class StoryGenerationResponse(BaseModel):
    message: str
    story_id: Optional[str] = None # If single story generation returns an ID
    # Add other relevant fields from run_generation_with if applicable

class BatchGenerationResponse(BaseModel):
    message: str
    num_stories_requested: int
    story_ids: List[str]

class ImageRegenerationRequest(BaseModel):
    for_characters: bool = False
    for_scenes: bool = False

# --- API Endpoints ---

@app.post("/story/generate", response_model=StoryGenerationResponse, tags=["Story Generation"])
async def generate_story_endpoint(payload: StoryGenerationRequest, approach: GenerationApproach = GenerationApproach.PROPOSED):
    """
    Generates a single story with the specified configuration and approach.
    Corresponds to `generate_story_with` from main.py.
    """
    try:
        validate_existing_plot(payload.existing_plot)
        validate_config(
            payload.min_num_choices, payload.max_num_choices,
            payload.min_num_choices_opportunity, payload.max_num_choices_opportunity,
            payload.num_chapters, payload.num_endings,
            payload.num_main_characters, payload.num_main_scenes
        )

        config = GenerationConfig(
            min_num_choices=payload.min_num_choices,
            max_num_choices=payload.max_num_choices,
            min_num_choices_opportunity=payload.min_num_choices_opportunity,
            max_num_choices_opportunity=payload.max_num_choices_opportunity,
            game_genre=payload.game_genre,
            themes=payload.themes,
            num_chapters=payload.num_chapters,
            num_endings=payload.num_endings,
            num_main_characters=payload.num_main_characters,
            num_main_scenes=payload.num_main_scenes,
            enable_image_generation=payload.enable_image_generation,
            existing_plot=payload.existing_plot,
            seed=payload.seed
        )
        logger.info(f"API - Generation config: {config}, Approach: {approach}")
        
        # Assuming run_generation_with returns a story object with an 'id' attribute
        # This part might need adjustment based on actual return type of run_generation_with
        story_result = run_generation_with(config, approach)
        story_id = getattr(story_result, 'id', None) # Safely get id

        return StoryGenerationResponse(message="Story generation initiated successfully.", story_id=story_id)

    except ValueError as e:
        logger.error(f"Validation error in /story/generate: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error during story generation in /story/generate")
        raise HTTPException(status_code=500, detail="Internal server error during story generation.")


@app.post("/story/batch-generate", response_model=BatchGenerationResponse, tags=["Story Generation"])
async def batch_generate_stories_endpoint(payload: BatchGenerationRequest):
    """
    Generates a batch of stories using the 'proposed' approach.
    Corresponds to `batch_generation` from main.py.
    """
    try:
        validate_config(
            payload.min_num_choices, payload.max_num_choices,
            payload.min_num_choices_opportunity, payload.max_num_choices_opportunity,
            payload.num_chapters, payload.num_endings,
            payload.num_main_characters, payload.num_main_scenes
        )

        config = GenerationConfig(
            min_num_choices=payload.min_num_choices,
            max_num_choices=payload.max_num_choices,
            min_num_choices_opportunity=payload.min_num_choices_opportunity,
            max_num_choices_opportunity=payload.max_num_choices_opportunity,
            game_genre=payload.game_genre,
            themes=payload.themes,
            num_chapters=payload.num_chapters,
            num_endings=payload.num_endings,
            num_main_characters=payload.num_main_characters,
            num_main_scenes=payload.num_main_scenes,
            enable_image_generation=payload.enable_image_generation,
            existing_plot=None, # Batch generation in main.py doesn't use existing_plot
            seed=payload.seed
        )
        logger.info(f"API - Batch generation config: {config}, N stories: {payload.n_stories}")
        
        generated_stories = run_batch_generation(config, payload.n_stories, GenerationApproach.PROPOSED)
        
        story_ids = [getattr(story, 'id', 'unknown_id') for story in generated_stories] if generated_stories else []
        logger.info(f"Finished generating {payload.n_stories} stories with proposed approach. Story ids: {story_ids}")
        
        return BatchGenerationResponse(
            message=f"Batch story generation for {payload.n_stories} stories initiated.",
            num_stories_requested=payload.n_stories,
            story_ids=story_ids
        )
    except ValueError as e:
        logger.error(f"Validation error in /story/batch-generate: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error during batch story generation in /story/batch-generate")
        raise HTTPException(status_code=500, detail="Internal server error during batch story generation.")


@app.get("/story/{story_id}/cost", tags=["Story Utilities"])
async def get_story_cost_endpoint(story_id: str):
    """Calculates and returns the estimated generation cost for a specific story."""
    try:
        cost_details = calculate_story_cost_logic(story_id)
        return cost_details
    except FileNotFoundError as e:
        logger.warning(f"File not found for story cost calculation {story_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Value error calculating cost for story {story_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error calculating cost for story {story_id}")
        raise HTTPException(status_code=500, detail=f"Error calculating cost: {str(e)}")


@app.get("/story/{story_id}/completion-time", tags=["Story Utilities"])
async def get_story_completion_time_endpoint(story_id: str):
    """Calculates and returns the time taken to generate a specific story."""
    try:
        time_details = get_story_completion_time_logic(story_id)
        return time_details
    except FileNotFoundError as e:
        logger.warning(f"File not found for story completion time {story_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Value error for story completion time {story_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error calculating completion time for story {story_id}")
        raise HTTPException(status_code=500, detail=f"Error calculating completion time: {str(e)}")


@app.post("/story/{story_id}/regenerate-images", tags=["Story Utilities"])
async def regenerate_story_images_endpoint(story_id: str, payload: ImageRegenerationRequest):
    """Regenerates images for a story's characters and/or scenes."""
    if not payload.for_characters and not payload.for_scenes:
        raise HTTPException(status_code=400, detail="Either for_characters or for_scenes must be true in the request body.")
    try:
        result = regenerate_story_images_logic(story_id, payload.for_characters, payload.for_scenes)
        return {"message": f"Image regeneration process for story {story_id} finished.", "details": result}
    except FileNotFoundError as e:
        logger.warning(f"File not found during image regeneration for {story_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Value error during image regeneration for {story_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))        
    except Exception as e:
        logger.exception(f"Error regenerating images for story {story_id}")
        raise HTTPException(status_code=500, detail=f"Error regenerating images: {str(e)}")


@app.delete("/story/{story_id}", status_code=200, tags=["Story Utilities"])
async def delete_story_data_endpoint(story_id: str):
    """Deletes a story and its associated data from the database."""
    try:
        delete_story_data_logic(story_id)
        return {"message": f"Story {story_id} and associated data deleted successfully."}
    except HTTPException: # Re-raise HTTPException if we set it above
        raise
    except Exception as e:
        logger.exception(f"Error deleting story {story_id}")
        raise HTTPException(status_code=500, detail=f"Error deleting story: {str(e)}")

# To run this (save as api.py in auto-vn-gen directory):
# Ensure you have fastapi and uvicorn installed: pip install fastapi uvicorn[standard]
# Run from terminal in auto-vn-gen directory: uvicorn api:app --reload --port 8000

if __name__ == "__main__":
    # This part is for direct execution (e.g., python api.py) which is not typical for uvicorn deployment
    # but can be useful for some quick tests if uvicorn is also started programmatically.
    # For standard deployment, use: uvicorn api:app --reload --port 8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 