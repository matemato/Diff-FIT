import json
import os
import time
from random import random

from PIL import Image
from tqdm import tqdm

from diff_fit.face_generation import FaceGeneration
from diff_fit.generation_utils.constants import MAX_SEED
from diff_fit.utils import get_model_id
from generate_images import GENERATED_IMAGES_DIR, GENERATED_IMG2IMG_DIR
from generate_images.config import ImageGenConfig, Img2ImgConfig


class GenerateImg2ImgImages:

    def __init__(self, gen_config: ImageGenConfig, img2img_config: Img2ImgConfig):
        self.gen_config = gen_config
        self.model_id = get_model_id(gen_config.model)
        self.face_generation = FaceGeneration(self.model_id)
        self.strength = img2img_config.strength
        self.use_prompt = img2img_config.use_prompt
        if self.use_prompt:
            dir = "prompt"
        else:
            dir = "no_prompt"

        self.number_of_images = img2img_config.number_of_images
        self.source_img_path = GENERATED_IMAGES_DIR / gen_config.data_path
        self.path = GENERATED_IMG2IMG_DIR / gen_config.data_path / dir
        os.makedirs(self.path, exist_ok=True)

    def generate_results(self) -> None:
        """
        Args:
            seed (int): Seed for random number generation.

        Generates an img2img image based on the initial image and saves it along with metadata.
        """
        for filename in sorted(os.listdir(self.source_img_path)):
            if self.gen_config.randomize_seed:
                seed = random.randint(1, MAX_SEED)
            else:
                seed = int(filename.split(".")[0])
            if filename.endswith(".png"):
                image_path = os.path.join(self.source_img_path, filename)
                json_path = os.path.join(
                    self.source_img_path, filename.replace(".png", ".json")
                )

                image = Image.open(image_path)

                if os.path.exists(json_path):
                    with open(json_path, "r") as file:
                        data = json.load(file)
                        prompt = data["prompt"] if self.use_prompt else ""
                        prompt = ", ".join(
                            prompt.split(", ")[:-3]
                        )  # Remove last 3 attributes
                        print(prompt)
                else:
                    return

                file_name = filename.split(".")[0]
                for i in range(self.number_of_images):
                    seed += i
                    start_time = time.time()
                    result, *_ = self.face_generation.img2img(
                        image_input=image,
                        prompt=prompt,
                        num_inference_steps=self.gen_config.steps,
                        guidance_scale=self.gen_config.cfg,
                        seed=seed,
                        strength=self.strength,
                        batch_count=1,
                        scheduler=self.gen_config.scheduler,
                    )
                    end_time = time.time()

                    os.makedirs(
                        f"{self.path}/{round(self.strength, 2)}/{file_name}",
                        exist_ok=True,
                    )

                    result[0].save(
                        f"{self.path}/{round(self.strength, 2)}/{file_name}/{i:03}.png"
                    )

                    self.save_metadata(
                        seed,
                        prompt,
                        file_name,
                        end_time - start_time,
                        i,
                    )

    def save_metadata(
        self, seed: int, prompt: str, file_name, time_taken: float, i: int
    ) -> None:
        """
        Args:
            seed (int): Seed for random number generation.
            prompt (str): The prompt used for image generation.
            file_name (str): Base name of the source image file.
            time_taken (float): Time taken for image generation.

        Saves the metadata of the generated image to a JSON file.
        """
        metadata = {
            "prompt": prompt,
            "seed": seed,
            "model": self.gen_config.model,
            "steps": self.gen_config.steps,
            "cfg": self.gen_config.cfg,
            "scheduler": self.gen_config.scheduler,
            "strength": self.strength,
            "time": time_taken,
        }

        with open(
            f"{self.path}/{round(self.strength, 2)}/{file_name}/{i:03}.json", "w"
        ) as file:
            json.dump(metadata, file, indent=4)


def main():
    """Main function to generate img2img images."""
    generate_img2img = GenerateImg2ImgImages(ImageGenConfig(), Img2ImgConfig())
    generate_img2img.generate_results()


if __name__ == "__main__":
    main()
