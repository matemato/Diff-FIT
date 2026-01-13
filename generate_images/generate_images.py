import json
import os
import random
import time

from tqdm import tqdm

from diff_fit.face_generation import FaceGeneration
from diff_fit.generation_utils.constants import GENERATION_DROPDOWN
from diff_fit.generation_utils.face_generation_utils import generate_prompt
from diff_fit.utils import get_model_id
from generate_images import GENERATED_IMAGES_DIR
from generate_images.config import ImageGenConfig


class GenerateImages:

    def __init__(self, config: ImageGenConfig):
        self.config = config
        self.model_id = get_model_id(config.model)
        self.face_generation = FaceGeneration(self.model_id)
        self.path = GENERATED_IMAGES_DIR / config.model
        os.makedirs(self.path, exist_ok=True)

    def get_random_attributes(self, seed: int) -> dict:
        """
        Args:
            seed (int): Seed for random number generation.

            Returns:
            dict: Dictionary of randomly selected face attributes.

        """
        random.seed(seed)
        age = random.choice(range(10, 80, 10))
        race = random.choice(GENERATION_DROPDOWN["Ethnicity"][1:])
        sex = random.choice(GENERATION_DROPDOWN["Sex"][1:])
        hair_length = random.choice(GENERATION_DROPDOWN["Hair length"][1:])
        hair_color = (
            random.choice(GENERATION_DROPDOWN["Hair color"][1:])
            if hair_length != "bald"
            else ""
        )
        hair_style = (
            random.choice(GENERATION_DROPDOWN["Hair style"][1:])
            if hair_length != "bald"
            else ""
        )
        eye_color = random.choice(GENERATION_DROPDOWN["Eye color"][1:])
        glasses = random.choice(GENERATION_DROPDOWN["Glasses"][1:])
        facial_hair = (
            random.choice(GENERATION_DROPDOWN["Facial hair"][1:])
            if sex == "male"
            else ""
        )
        return {
            "age": age,
            "race": race,
            "sex": sex,
            "hair_length": hair_length,
            "hair_color": hair_color,
            "hair_style": hair_style,
            "eye_color": eye_color,
            "glasses": glasses,
            "facial_hair": facial_hair,
        }

    def generate_results(self, seed: int, attrs: dict) -> None:
        """
        Args:
            seed (int): Seed for random number generation.
            attrs (dict): Dictionary of face attributes.

        Generates an image based on the provided attributes and saves it along with metadata.
        """

        prompt, negative_prompt = generate_prompt(
            ("", *attrs.values()),
            seed=seed,
            randomize=self.config.randomize,
            additional_negative_prompt=True,
        )

        start_time = time.time()
        image, *_ = self.face_generation.generate_images(
            pipe=None,
            initial_prompt=prompt,
            num_inference_steps=self.config.steps,
            guidance_scale=self.config.cfg,
            batch_count=1,
            strength=None,
            init_image=None,
            mask=None,
            randomize=self.config.randomize,
            negative_prompt=negative_prompt,
            seed=seed,
            scheduler=self.config.scheduler,
        )
        end_time = time.time()

        image[0].save(f"{self.path}/{seed:04}.png")

        self.save_metadata(
            attrs,
            seed,
            prompt,
            negative_prompt,
            end_time - start_time,
        )

    def save_metadata(
        self, attrs: dict, seed: int, prompt: str, negative_prompt: str, time: float
    ) -> None:
        """
        Args:
            attrs (dict): Dictionary of face attributes.
            seed (int): Seed for random number generation.
            prompt (str): The prompt used for image generation.
            negative_prompt (str): The negative prompt used for image generation.
            time (float): Time taken for image generation.

        Saves the metadata of the generated image to a text file.
        """
        attrs["prompt"] = prompt
        attrs["negative_prompt"] = negative_prompt
        attrs["steps"] = self.config.steps
        attrs["cfg"] = self.config.cfg
        attrs["scheduler"] = self.config.scheduler
        attrs["time"] = time

        with open(f"{self.path}/{seed:04}.json", "w") as file:
            json.dump(attrs, file, indent=4)


def main():
    """Main function to generate images based on random attributes."""
    generate_images = GenerateImages(ImageGenConfig())
    for i in tqdm(range(generate_images.config.number_of_images), "Generating Images"):
        attrs = generate_images.get_random_attributes(i)
        generate_images.generate_results(i, attrs)


if __name__ == "__main__":
    main()
