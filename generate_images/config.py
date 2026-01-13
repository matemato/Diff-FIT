from dataclasses import dataclass


@dataclass
class ImageGenConfig:
    """
    Configuration for image generation.
    """

    model: str = "lightning5"
    steps: int = 5
    cfg: float = 1
    randomize: bool = True
    scheduler: str = "single"
    number_of_images: int = 20


@dataclass
class InpaintingConfig:
    """
    Configuration for inpainting tasks.
    """

    mask_attribute: str = "Beard"
    face_attribute: str = "facial_hair"
    adjective: str = "beard"
    transformed_adjective: str = "no facial hair"
    strength: float = 0.8


@dataclass
class Img2ImgConfig:
    """
    Configuration for img2img generation.
    """

    strength: float = 1
    number_of_images: int = 2
    use_prompt_file: bool = True
