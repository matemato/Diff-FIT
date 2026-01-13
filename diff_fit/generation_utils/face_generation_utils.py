import random

import torch
from diffusers import (
    AutoencoderKL,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    TCDScheduler,
)

import diff_fit.generation_utils.constants as const


def get_txt2img_pipeline(model_id, device):
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    ).to("cuda")
    return AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        vae=vae,
    ).to(device)


def get_img2img_pipeline(pipeline, device):
    return AutoPipelineForImage2Image.from_pipe(pipeline).to(device)


def get_inpainting_pipeline(pipeline, device):
    return AutoPipelineForInpainting.from_pipe(pipeline).to(device)


def get_scheduler(pipe=None, scheduler=None, original_scheduler_config=None):
    schedulers = {
        "single": DPMSolverSinglestepScheduler.from_config(
            original_scheduler_config,
            lower_order_final=True,
        ),
        "single_karras": DPMSolverSinglestepScheduler.from_config(
            original_scheduler_config, use_karras_sigmas=True
        ),
        "multi": DPMSolverMultistepScheduler.from_config(
            original_scheduler_config,
        ),
        "multi_karras": DPMSolverMultistepScheduler.from_config(
            original_scheduler_config, use_karras_sigmas=True
        ),
        "TCD": TCDScheduler.from_config(original_scheduler_config),
        "Euler": EulerDiscreteScheduler.from_config(original_scheduler_config),
        "sde": DPMSolverSDEScheduler.from_config(original_scheduler_config),
        "sde_karras": DPMSolverSDEScheduler.from_config(
            original_scheduler_config, use_karras_sigmas=True
        ),
        "sde_my_way": DPMSolverSDEScheduler(),
        "sde_karras_my_way": DPMSolverSDEScheduler(use_karras_sigmas=True),
    }

    if scheduler is None:
        return pipe.scheduler
    elif scheduler == "random":
        random_scheduler = random.choice(list(schedulers.values()))
        return random_scheduler

    return schedulers[scheduler]


def generate_prompt(prompt, seed=-1, randomize=True, additional_negative_prompt=False):
    if seed == -1:
        seed = random.randint(1, const.MAX_SEED)
    random.seed(seed)
    (
        age,
        race,
        sex,
        hair_length,
        hair_color,
        hair_style,
        eye_color,
        glasses,
        facial_hair,
        negative_prompt,
    ) = (
        randomize_prompt(prompt[1:]) if randomize else prompt[1:] + ("",)
    )

    generated_prompt = f"{age} year old {race} {sex}"

    if (
        hair_color != "" or hair_length != "" or hair_style != ""
    ) and hair_length != "bald":
        generated_prompt += " with "
        if hair_color != "":
            generated_prompt += f"{hair_color} "
        if hair_length != "":
            generated_prompt += f"{hair_length} "
        if hair_style != "":
            generated_prompt += f"{hair_style} "
        generated_prompt += "hair"

    if hair_length == "bald":
        generated_prompt += ", bald"

    if eye_color != "":
        generated_prompt += f", {eye_color} eyes"

    if glasses != "":
        generated_prompt += f", {glasses}"

    if facial_hair != "" and sex != "female":
        generated_prompt += f", {facial_hair}"

    if prompt[0] != "":
        generated_prompt += ", " + prompt[0]

    if randomize:
        generated_prompt += f", from {random.choice(const.EUROPEAN_COUNTRIES)}"

    generated_prompt += ", portrait, front shot, white background"

    ## remove multiple spaces
    generated_prompt = " ".join(generated_prompt.split())

    if additional_negative_prompt:
        negative_prompt = (
            f"{const.NEGATIVE_PROMPT}, {negative_prompt}"
            if negative_prompt != ""
            else const.NEGATIVE_PROMPT
        )
    if negative_prompt == "":
        negative_prompt = None

    return generated_prompt, negative_prompt


def randomize_prompt(prompt, seed=-1):
    if seed == -1:
        seed = random.randint(1, const.MAX_SEED)
    random.seed(seed)
    (
        age,
        race,
        sex,
        hair_length,
        hair_color,
        hair_style,
        eye_color,
        glasses,
        facial_hair,
    ) = prompt
    negative_prompt = []

    age += random.randint(-10, 10)

    if age < 5:
        age = 5

    if sex == "":
        sex = random.choice(const.GENERATION_DROPDOWN["Sex"])

    if hair_color == "":
        hair_color = random.choice(const.GENERATION_DROPDOWN["Hair color"])
    if hair_color != "" and hair_color in const.HAIR_COLORS.keys():
        hair_color = random.choice(const.HAIR_COLORS[hair_color])

    if hair_length == "":
        hair_length = random.choice(const.GENERATION_DROPDOWN["Hair length"])
    if hair_length != "" and hair_length in const.HAIR_LENGTHS.keys():
        hair_length = random.choice(const.HAIR_LENGTHS[hair_length])

    if hair_style == "":
        hair_style = random.choice(const.GENERATION_DROPDOWN["Hair style"])
    if hair_style != "" and hair_style in const.HAIR_STYLES.keys():
        hair_style = random.choice(const.HAIR_STYLES[hair_style])

    if eye_color == "":
        eye_color = random.choice(const.GENERATION_DROPDOWN["Eye color"])
    if eye_color != "" and eye_color in const.EYE_COLORS.keys():
        eye_color = random.choice(const.EYE_COLORS[eye_color])

    if glasses == "":
        glasses = random.choice(const.GENERATION_DROPDOWN["Glasses"])
    if glasses != "" and glasses in const.GLASSES.keys():
        glasses = random.choice(const.GLASSES[glasses])
    elif glasses == "no glasses":
        negative_prompt.append("glasses, eyeglasses, sunglasses")

    if facial_hair == "":
        facial_hair = random.choice(const.GENERATION_DROPDOWN["Facial hair"])

    if facial_hair == "mustache":
        negative_prompt.append("beard")
    elif facial_hair == "beard":
        negative_prompt.append("mustache")
    elif facial_hair == "no facial hair":
        negative_prompt.append("beard, mustache, facial hair")

    if facial_hair != "" and facial_hair in const.FACIAL_HAIR.keys():
        facial_hair = random.choice(const.FACIAL_HAIR[facial_hair])

    return (
        age,
        race,
        sex,
        hair_length,
        hair_color,
        hair_style,
        eye_color,
        glasses,
        facial_hair,
        ", ".join(negative_prompt),
    )
