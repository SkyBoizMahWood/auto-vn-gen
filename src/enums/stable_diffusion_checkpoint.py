from enum import Enum

class StableDiffusionCheckpoint(Enum):
    NOVA_ANIME_XL = "novaAnimeXL_ilV50"
    CRYSTAL_CLEAR_XL = "crystalClearXL_ccxl"
    
    @property
    def filename(self) -> str:
        return f"{self.value}.safetensors"