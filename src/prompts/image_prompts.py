from src.models.story.character_data import CharacterData
from src.models.story.scene_data import SceneData


def get_negative_image_prompt() -> str:
    return ("multiple people, poorly Rendered face, poorly drawn face, poor facial details, poorly drawn hands, "
            "poorly rendered hands, low resolution, blurry image, oversaturated, bad anatomy, signature, watermark, "
            "username, error, out of frame, extra fingers, mutated hands, poorly drawn hands, malformed limbs, "
            "missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, "
            "long neck, bad face, text, realistic")


def get_character_image_prompt(character: CharacterData) -> str:
    prompt_template = (
        "A close-up face portrait image game asset of a 2D character artwork in a classic RPG game on "
        "a plain white background. {name} is a {gender} {species} who is {age} years old. "
        "They have {physical_appearance}. No text. One image only. Front face only. Centered. No drawings. "
        "Anime-style asset. Detailed face.")

    prompt = prompt_template.format(
        name=character.first_name + " " + character.last_name,
        species=character.species,
        gender=character.gender,
        age=character.age,
        physical_appearance=" and ".join(character.physical_appearance) if isinstance(character.physical_appearance, list) and all(isinstance(item, str) for item in character.physical_appearance) else character.physical_appearance
    )

    return prompt


def get_scene_image_prompt(scene: SceneData) -> str:
    prompt_template = (
        "An image of a 2D scene artwork in a classic RPG game with a landscape scene background. "
        "This is a scene of {title} located in {location}. The scene is {description}. No text. "
        "One image only. Centered. No drawings. Anime-style asset.")

    prompt = prompt_template.format(
        title=scene.title,
        location=scene.location,
        description=scene.description
    )

    return prompt

def get_scene_negative_image_prompt() -> str:
    negative_prompt = (
        "EasyNegativeV2, worst quality, low quality, medium quality, "
        "deleted, lowres, comic, bad anatomy, bad hands, text, error, "
        "missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, "
        "signature, watermark, username, blurry, less than 5 fingers, "
        "more than 5 fingers, bad hands, bad hand anatomy, missing fingers, "
        "extra fingers, mutated hands, disfigured hands, deformed hands, "
        "(double eyebrows:1.3), deformed lips, bad teeth, deformed teeth, "
        "(multiple tails:1.1)"
    )
    return negative_prompt