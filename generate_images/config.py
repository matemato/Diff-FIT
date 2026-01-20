from dataclasses import dataclass


@dataclass
class ImageGenConfig:
    """
    Configuration for image generation.
    """

    model: str = "lightning5"
    steps: int = 5
    cfg: float = 1
    randomize_variations: bool = False
    randomize_seed: bool = False
    scheduler: str = "single"
    number_of_images: int = 10
    data_path: str = "demo_images"


@dataclass
class InpaintingConfig:
    """
    Configuration for inpainting tasks.
    """

    mask_attribute: str = "Hair"
    face_attribute: str = "hair_color"
    adjective: str = "blonde"
    transformed_adjective: str = "dark brown"
    strength: float = 4 / 5


@dataclass
class Img2ImgConfig:
    """
    Configuration for img2img generation.
    """

    strength: float = 4 / 5
    number_of_images: int = 1
    use_prompt: bool = False
