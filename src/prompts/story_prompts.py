import random

from src.models.generation_config import GenerationConfig
from src.models.story.story_choice import StoryChoice
from src.models.story_chunk import StoryChunk
from src.models.story_data import StoryData

JSON_MAGIC_PHRASE = "Return output in JSON format and only the JSON in the Markdown code block. JSON."


def get_plot_prompt(config: GenerationConfig) -> str:
    if config.themes is None or len(config.themes) == 0:
        config.themes = ["sci-fi", "fantasy", "middle-age", "utopia", "mythical creatures", "world scale"]

    if config.num_main_characters is None:
        config.num_main_characters = random.randint(3, 7)

    if config.num_main_scenes is None:
        config.num_main_scenes = random.randint(1, 5)

    if config.num_chapters is None:
        config.num_chapters = random.randint(3, 7)

    return f"""Write a game story synopsis. Then generate 1 story beginning, {config.num_endings} possible endings, {config.num_main_characters} main characters, and {config.num_main_scenes} main scenes. There are a total of {config.num_chapters} chapters. {JSON_MAGIC_PHRASE}

# Output format
{{
"title": game title,
"genre": game genre,
"themes": [words],
"main_scenes": [{{"id": id (integer), "title": location name, "location": where is this place, "description": describe location}}].
"main_characters": [{{"id": id (integer), "first_name": first name, "last_name": last name ("" if unknown), "species": species, "age": exact age or description, "gender": gender of the character, "role": role of the character, "background": background story, "place_of_birth": location, "physical_appearance": [list of details]}}]
"synopsis": synopsis,
"chapter_synopses": [{{"chapter": chapter_number (integer), "synopsis": synopsis, "character_ids": id of featured characters in this chapter, "scene_ids": id of featured scenes in this chapter}}]
"beginning": beginning of the story,
"endings": [{{"id": id (integer), "ending": ending}}]
}}

# Game information
Game genre: {config.game_genre}
Themes: {config.get_themes_str()}"""


def get_story_until_choices_opportunity_prompt(config: GenerationConfig, story_data: StoryData, num_choices: int,
                                               used_opportunity: int, chapter: int) -> str:
    return f"""Generate possible narratives and dialogues for a {config.game_genre} game, culminating in {num_choices} choices that the player can make to influence the course of the story. {JSON_MAGIC_PHRASE}

# Current chapter
{chapter} of {config.num_chapters} chapters

# Current chapter synopsis
{story_data.chapter_synopses[chapter - 1].synopsis}

# Choice opportunities
Currently used: {used_opportunity} out of {config.max_num_choices_opportunity} for this chapter

# Output format
{{
"id": id (integer),
"story_so_far": story so far,
"story": [{{"id": id signifies the order (integer), "speaker": speaker name or "Narration", "speaker_id": id of the speaking character or -1 for narration, "scene_title": title of a current scene (always one of the scenes in the first story data), "scene_id": id of the current scene (always one of the scenes in the first story data), "text": dialogue or narration}}],
"choices": [{{"id": id (integer), "choice": choice, "description": description}}]
}}"""


def get_story_based_on_selected_choice_prompt(config: GenerationConfig, story_data: StoryData,
                                              selected_choice: StoryChoice,
                                              num_choices: int, used_opportunity: int, chapter: int) -> str:
    return f"""Generate possible narratives and dialogues for a {config.game_genre} game, based on the selected choice with {num_choices} possible choices. {JSON_MAGIC_PHRASE}

# Output format
{{
"id": id (integer),
"story_so_far": story so far,
"story": [{{"id": id signifies the order (integer), "speaker": speaker name or "Narration", "speaker_id": id of the speaking character or -1 for narration, "scene_title": title of a current scene (always one of the scenes in the first story data), "scene_id": id of the current scene (always one of the scenes in the first story data), "text": dialogue or narration}}],
"choices": [{{"id": id (integer), "choice": choice, "description": description}}]
}}

# Current chapter
{chapter} of {config.num_chapters} chapters

# Current chapter synopsis
{story_data.chapter_synopses[chapter - 1].synopsis}

# Choice opportunities
Currently used: {used_opportunity} out of {config.max_num_choices_opportunity} for this chapter

# The selected choice
{selected_choice}"""


def get_story_until_chapter_end_prompt(config: GenerationConfig, story_data: StoryData, story_chunk: StoryChunk) -> str:
    return f"""Generate possible narratives and dialogues for a {config.game_genre} game, culminating in the end of the chapter. {JSON_MAGIC_PHRASE}

# Output format
{{
"id": id (integer),
"story_so_far": story so far,
"story": [{{"id": id signifies the order (integer), "speaker": speaker name or "Narration", "speaker_id": id of the speaking character or -1 for narration, "scene_title": title of a current scene (always one of the scenes in the first story data), "scene_id": id of the current scene (always one of the scenes in the first story data), "text": dialogue or narration}}]
}}

# The story so far
{story_chunk.story_so_far}

# Next chapter synopsis
{story_data.chapter_synopses[story_chunk.chapter].synopsis}"""


def get_story_until_game_end_prompt(config: GenerationConfig, story_data: StoryData, story_chunk: StoryChunk) -> str:
    selected_ending_idx = random.randint(0, config.num_endings - 1)
    return f"""Generate possible narratives and dialogues for a {config.game_genre} game, culminating in the end of the game. {JSON_MAGIC_PHRASE}

# Output format
{{
"id": id (integer),
"story_so_far": story so far,
"story": [{{"id": id signifies the order (integer), "speaker": speaker name or "Narration", "speaker_id": id of the speaking character or -1 for narration, "scene_title": title of a current scene (always one of the scenes in the first story data), "scene_id": id of the current scene (always one of the scenes in the first story data), "text": dialogue or narration}}],
}}

# The story so far
{story_chunk.story_so_far}

# The selected ending
{story_data.endings[selected_ending_idx]}"""
