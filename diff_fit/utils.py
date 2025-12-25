import argparse
import math
import os

import gdown  # type: ignore
import gradio as gr  # type: ignore
import pandas as pd  # type: ignore
from huggingface_hub import snapshot_download  # type: ignore
from PIL import Image

import diff_fit.generation_utils.constants as generation_const
from diff_fit import DATA_DIR, WEIGHTS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="1.5 - stable_diffusion1.5, realVis - Realistic_Vision, XL - stable_diffusion_XL, turbo - RealVis_Turbo, lightning - RealVis_Lightning",
        default="lightning5",
    )
    parser.add_argument("--port", default=generation_const.SERVER_PORT, type=int)

    return parser.parse_args()


def get_negative_prompt(use):
    return generation_const.NEGATIVE_PROMPT if use else None


def get_model_id(name):
    if name == "1.5":
        return "runwayml/stable-diffusion-v1-5"
    elif name == "realVis":
        return "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    elif name == "XL":
        return "stabilityai/stable-diffusion-xl-base-1.0"
    elif name == "turbo":
        return "SG161222/RealVisXL_V3.0_Turbo"
    elif name == "lightning4":
        return "SG161222/RealVisXL_V4.0_Lightning"
    elif name == "lightning5":
        return "SG161222/RealVisXL_V5.0_Lightning"
    elif name == "flux":
        return "black-forest-labs/FLUX.1-schnell"


def create_image_grid(images):
    # Calculate the size of each square in the grid

    col_size = math.ceil(math.sqrt(len(images)))
    row_size = math.ceil(len(images) / col_size)

    # Calculate the size of the entire grid
    grid_width = col_size * images[0].width
    grid_height = row_size * images[0].height

    # Create a new blank image with the calculated size
    grid_image = Image.new("RGB", (grid_width, grid_height), color="white")

    # Paste each image into its corresponding square
    for i, img in enumerate(images):
        row = i // col_size
        col = i % col_size
        x = col * img.width
        y = row * img.height
        grid_image.paste(img, (x, y))

    return grid_image


def reset_sliders(*offsets):
    return [0] * len(offsets)


def get_inpainting_tab_state(face_feature: str):
    return face_feature


def update_history(new_image, history, index=None):
    if history is None:
        history = []
    if index is None:
        history.append(new_image)
    else:
        # history.append(Image.open(new_image[index][0]))
        history.append(new_image[index][0])
    return history


def use_this_output(
    image,
    number_of_outputs: int,
    index: int,
):
    if index == -1:
        return [image[1]] * number_of_outputs
    return [image[index][0]] * number_of_outputs


def reset_index():
    return 0


def update_selection(selection: gr.SelectData):
    return selection.index


def update_text(text):
    return text


def download_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(f"{WEIGHTS_DIR}/lightning_drag/", exist_ok=True)
    os.makedirs(f"{WEIGHTS_DIR}/resnet/", exist_ok=True)
    os.makedirs(f"{WEIGHTS_DIR}/face_landmarks/", exist_ok=True)

    ## lightning drag weights
    models = [
        "Lykon/dreamshaper-8-inpainting",
        "latent-consistency/lcm-lora-sdv1-5",
        "h94/IP-Adapter",
        "stabilityai/sd-vae-ft-mse",
        "LightningDrag/lightning-drag-sd15",
    ]

    for model in models:
        print("Downloading", model)
        snapshot_download(
            repo_id=model,
            local_dir=f"{WEIGHTS_DIR}/lightning_drag/{model.split('/')[-1]}/",
        )

    ## resnet weights
    print("Downloading resnet weights")
    gdown.download(
        "https://drive.google.com/uc?id=1E0mrvluHEbfKi5zYOfat7Eb4slPmkU6P",
        output=f"{WEIGHTS_DIR}/resnet/resnet18.pt",
    )

    ## face landmark weights
    print("Downloading face landmark weights")
    gdown.download(
        "https://drive.google.com/uc?id=1mhYOvEYUEak33l_eDiYES2Q9gPdPCO8I",
        output=f"{WEIGHTS_DIR}/face_landmarks/shape_predictor_68_face_landmarks.dat",
    )


def save_file(
    save_dir,
    pipeline,
    scheduler,
    karras,
    seed,
    prompt,
    negative_prompt,
    steps,
    cfg,
    strength,
    points,
    mask,
    transformation,
    result,
    crop_inpaint,
):
    if save_dir is "" or save_dir is None:
        return
    print(f"Saving to {save_dir}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_DIR / "target_images", exist_ok=True)
    path = f"{DATA_DIR}/target_images/{save_dir}"

    columns = [
        "pipeline",
        "scheduler",
        "karras_sigmas",
        "seed",
        "prompt",
        "negative_prompt",
        "steps",
        "cfg",
        "strength",
        "crop_inpaint",
        "points",
        "mask",
        "transformation",
        "result",
    ]

    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(f"{path}/images")
        os.makedirs(f"{path}/masks")
        os.makedirs(f"{path}/transformations")
        df = pd.DataFrame(
            columns=columns,
        )
        df.to_csv(f"{path}/history.csv", index=True)

    df = pd.read_csv(f"{path}/history.csv", index_col=0)

    try:
        new_step = df.index[-1] + 1
    except Exception as e:
        print(e)
        new_step = 0

    if mask is not None:
        mask_path = f"masks/{new_step}.png"
        mask.save(f"{path}/{mask_path}")
    else:
        mask_path = None

    if transformation is not None:
        transformation_path = f"transformations/{new_step}.png"
        transformation.save(f"{path}/{transformation_path}")
    else:
        transformation_path = None

    if result is not None:
        result_path = f"images/{new_step}.png"
        result.save(f"{path}/{result_path}")
    else:
        result_path = None

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "pipeline": pipeline,
                    "scheduler": scheduler,
                    "karras_sigmas": karras,
                    "seed": seed,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "cfg": cfg,
                    "strength": strength,
                    "crop_inpaint": (
                        crop_inpaint
                        if pipeline == "StableDiffusionXLInpaintPipeline"
                        else None
                    ),
                    "points": points,
                    "mask": mask_path,
                    "transformation": transformation_path,
                    "result": result_path,
                },
                index=[new_step],
            ),
        ],
        ignore_index=False,
    )

    df.to_csv(f"{path}/history.csv", index=True)
