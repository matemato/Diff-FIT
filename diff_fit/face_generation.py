import random
from copy import deepcopy

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

import diff_fit.generation_utils.constants as generation_const
from diff_fit.generation_utils.face_generation_utils import (
    generate_prompt,
    get_img2img_pipeline,
    get_inpainting_pipeline,
    get_scheduler,
    get_txt2img_pipeline,
)
from diff_fit.inpaint_utils.utils import crop_mask, fix_inpaint_prompt
from diff_fit.utils import create_image_grid


class FaceGeneration:
    def __init__(self, model_id):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.txt2img_pipe = get_txt2img_pipeline(model_id, self.device)
        self.img2img_pipe = get_img2img_pipeline(self.txt2img_pipe, self.device)
        self.inpaint_pipe = get_inpainting_pipeline(self.txt2img_pipe, self.device)
        self.original_scheduler = deepcopy(self.txt2img_pipe.scheduler.config)
        self.txt2img_pipe.scheduler = get_scheduler(
            pipe=self.txt2img_pipe, original_scheduler_config=self.original_scheduler
        )
        self.img2img_pipe.scheduler = get_scheduler(
            pipe=self.img2img_pipe, original_scheduler_config=self.original_scheduler
        )
        self.inpaint_pipe.scheduler = get_scheduler(
            pipe=self.inpaint_pipe, original_scheduler_config=self.original_scheduler
        )
        self.three = 0

    def generate_images(
        self,
        pipe,
        initial_prompt,
        num_inference_steps,
        guidance_scale,
        batch_count,
        batch_size=1,
        strength=None,
        init_image=None,
        mask=None,
        randomize=True,
        negative_prompt="",
        seed=-1,
        scheduler=None,
    ):
        #### ONLY FOR GENERATING IMAGES TODO
        if pipe is None:
            pipe = self.txt2img_pipe

        self.three += 1
        randomize_seed = seed == -1

        images = []
        for i in range(batch_count):
            if scheduler is not None:
                pipe.scheduler = get_scheduler(pipe, scheduler, self.original_scheduler)

            if randomize_seed:
                seed = random.randint(1, generation_const.MAX_SEED)
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            print(f"Generating batch {i+1}/{batch_count} with seed {seed}")
            if not isinstance(initial_prompt, str):
                prompt, negative_prompt = generate_prompt(
                    initial_prompt,
                    seed=seed,
                    randomize=randomize,
                    additional_negative_prompt=True,
                )
            else:
                prompt = initial_prompt
            if negative_prompt == "":
                negative_prompt = None
            print(f"Prompt: {prompt}")
            print(f"Negative prompt: {negative_prompt}")
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                device="cuda",
                do_classifier_free_guidance=True,
            )

            imgs = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=batch_size,
                strength=strength,
                image=init_image,
                mask_image=mask,
                generator=generator,
                guidance_rescale=0.7,
            ).images

            for im in imgs:
                images.append(im)

            if not randomize_seed:
                seed += 1

        if len(images) > 1:
            images.insert(0, create_image_grid(images))

        return (
            images,
            pipe.__class__.__name__,
            pipe.scheduler.__class__.__name__,
            pipe.scheduler.use_karras_sigmas,
            seed,
            prompt,
            negative_prompt,
            num_inference_steps,
            guidance_scale,
            strength,
            None,
            mask,
            None,
            images[0],
        )

    def txt2image(
        self,
        num_inference_steps,
        guidance_scale,
        batch_count,
        randomize,
        seed,
        scheduler,
        description_only=False,
        *prompt,
    ):
        if description_only:
            prompt = prompt[0]
        return self.generate_images(
            pipe=self.txt2img_pipe,
            initial_prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            batch_count=batch_count,
            randomize=randomize,
            seed=seed,
            scheduler=scheduler,
        )

    def img2img(
        self,
        image_input,
        prompt,
        num_inference_steps,
        seed,
        guidance_scale,
        strength,
        batch_count,
        scheduler,
    ):
        return self.generate_images(
            pipe=self.img2img_pipe,
            initial_prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            batch_count=batch_count,
            strength=strength,
            init_image=image_input,
            seed=seed,
            randomize=False,
            scheduler=scheduler,
        )

    def inpaint(
        self,
        inpaint_canvas,
        prompt,
        num_inference_steps,
        seed,
        guidance_scale,
        strength,
        batch_count,
        scheduler,
        crop_inpaint=False,
        paste_img_back=False,
        mask=None,
        adjective=None,
        facial_feature=None,
        negative_prompt=None,
    ):
        if mask is not None:
            if prompt == "":
                prompt, negative_prompt = fix_inpaint_prompt(adjective, facial_feature)
            init_image = inpaint_canvas
        else:
            init_image, mask = inpaint_canvas["background"], inpaint_canvas["layers"][0]

            init_image = init_image.convert("RGB")
            mask = mask.convert("RGB")

        if crop_inpaint:
            initial_image = init_image
            init_image, mask, regions, cropped_shape = crop_mask(
                np.array(init_image), np.array(mask)[:, :, 0]
            )

        inpaint_results = self.generate_images(
            pipe=self.inpaint_pipe,
            initial_prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            batch_count=batch_count,
            seed=seed,
            strength=strength,
            init_image=init_image,
            mask=mask,
            scheduler=scheduler,
        )

        for i, img in enumerate(inpaint_results[0]):
            if crop_inpaint:
                if len(inpaint_results[0]) > 1 and i == 0:
                    continue
                img = np.array(img)
                img = cv2.resize(img, cropped_shape, interpolation=cv2.INTER_AREA)
                initial_image = np.array(initial_image)
                initial_image[
                    regions[0] : regions[1] + 1, regions[2] : regions[3] + 1
                ] = img
                inpaint_results[0][i] = Image.fromarray(initial_image).convert("RGB")
            elif paste_img_back:
                if img.size == init_image.size:
                    mask = mask.convert("L")
                    for _ in range(3):
                        mask = mask.filter(ImageFilter.MaxFilter(3))
                    # mask.save("mask_after.png")
                    binary_mask = mask.point(lambda x: 255 if x > 127 else 0)

                    binary_mask = binary_mask.convert("1")
                    img = Image.composite(img, init_image, binary_mask)

                inpaint_results[0][i] = img

        return inpaint_results

    def predefined_inpaint(
        self,
        init_image,
        mask,
        adjective,
        facial_feature,
        num_inference_steps,
        seed,
        guidance_scale,
        strength,
        batch_count,
        scheduler,
    ):
        prompt, negative_prompt = fix_inpaint_prompt(adjective, facial_feature)

        images = self.generate_images(
            pipe=self.inpaint_pipe,
            initial_prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            batch_count=batch_count,
            seed=seed,
            strength=strength,
            init_image=init_image,
            mask=mask,
            scheduler=scheduler,
        )

        output_images = []

        for image in images:
            if image.size == init_image.size:
                binary_mask = mask.point(lambda x: 255 if x > 127 else 0)

                binary_mask = binary_mask.convert("1")

                image = Image.composite(image, init_image, binary_mask)

            output_images.append(image)

        return output_images
