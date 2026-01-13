import json
import os
import time

from PIL import Image

from diff_fit.face_generation import FaceGeneration
from diff_fit.face_inpaint import FaceInpaint
from diff_fit.utils import get_model_id
from generate_images import GENERATED_IMAGES_DIR, GENERATED_INPAINTING_DIR
from generate_images.config import ImageGenConfig, InpaintingConfig


class GenerateInpaintingResults:
    def __init__(self, gen_config: ImageGenConfig, inpaint_config: InpaintingConfig):
        self.gen_config = gen_config
        self.mask_attribute = inpaint_config.mask_attribute
        self.face_attribute = inpaint_config.face_attribute
        self.adjective = inpaint_config.adjective
        self.transformed_adjective = inpaint_config.transformed_adjective
        self.strength = inpaint_config.strength
        self.model_id = get_model_id(gen_config.model)
        self.face_generation = FaceGeneration(self.model_id)
        self.face_inpaint = FaceInpaint()
        self.source_img_path = GENERATED_IMAGES_DIR / gen_config.model
        self.save_path = (
            GENERATED_INPAINTING_DIR
            / gen_config.model
            / self.face_attribute
            / f"{self.adjective}_to_{self.transformed_adjective}"
        )
        os.makedirs(self.save_path, exist_ok=True)

    def generate_results(self) -> None:
        """
        Generates inpainting results based on previously generated images.
        Saves the results and their metadata.
        """
        print(
            f"\nGenerating inpainting images for {self.face_attribute} - {self.adjective} to {self.transformed_adjective}...\n"
        )
        for filename in sorted(os.listdir(self.source_img_path)):
            if filename.endswith(".png"):
                image_path = os.path.join(self.source_img_path, filename)
                json_path = os.path.join(
                    self.source_img_path, filename.replace(".png", ".json")
                )

                image = Image.open(image_path)

                if os.path.exists(json_path):
                    with open(json_path, "r") as file:
                        data = json.load(file)
                        if self.adjective not in data[self.face_attribute]:
                            continue
                else:
                    return

                self.face_inpaint.init_inpaint(image)
                mask, _ = self.face_inpaint.show_mask(image, self.mask_attribute)

                start_time = time.time()
                result = self.face_generation.inpaint(
                    inpaint_canvas=image,
                    prompt="",
                    num_inference_steps=self.gen_config.steps,
                    seed=int(filename.split(".")[0]),
                    guidance_scale=self.gen_config.cfg,
                    strength=self.strength,
                    batch_count=1,
                    scheduler=self.gen_config.scheduler,
                    mask=mask,
                    adjective=self.transformed_adjective,
                    facial_feature=self.mask_attribute,
                    paste_img_back=True,
                )[0]
                end_time = time.time()

                result[0].save(f"{self.save_path}/{filename}")

                self.save_metadata(int(filename.split(".")[0]), end_time - start_time)

    def save_metadata(self, seed: int, time: float) -> None:
        """
        Args:
            seed (int): Seed for random number generation.
            time (float): Time taken for inpainting.

        Saves the metadata of the inpainted image to a JSON file.
        """
        metadata = {
            "mask_attribute": self.mask_attribute,
            "face_attribute": self.face_attribute,
            "adjective": self.adjective,
            "transformed_adjective": self.transformed_adjective,
            "model": self.gen_config.model,
            "steps": self.gen_config.steps,
            "cfg": self.gen_config.cfg,
            "scheduler": self.gen_config.scheduler,
            "strength": self.strength,
            "time": time,
        }

        with open(f"{self.save_path}/{seed:04}.json", "w") as file:
            json.dump(metadata, file, indent=4)


def main() -> None:
    """Main function to generate inpainting results based on generated images."""
    generate_inpainting = GenerateInpaintingResults(
        ImageGenConfig(), InpaintingConfig()
    )
    generate_inpainting.generate_results()


if __name__ == "__main__":
    main()
