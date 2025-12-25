import cv2
import numpy as np
from PIL import Image

from diff_fit.generation_utils.constants import NEGATIVE_PROMPT


def crop_mask(image, mask, padding=50):
    # Find the coordinates of non-zero (255) values
    rows, cols = np.where(mask == 255)

    # Get bounding box
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Apply padding
    height = max_row - min_row + 1
    width = max_col - min_col + 1

    # Determine the additional padding needed to make it square
    if height > width:
        extra_padding = (height - width) // 2
        min_col = max(min_col - extra_padding, 0)
        max_col = min(max_col + extra_padding + (height - width) % 2, mask.shape[1] - 1)
    elif width > height:
        extra_padding = (width - height) // 2
        min_row = max(min_row - extra_padding, 0)
        max_row = min(max_row + extra_padding + (width - height) % 2, mask.shape[0] - 1)

    # Apply the original padding
    min_row = max(min_row - padding, 0)
    max_row = min(max_row + padding, mask.shape[0] - 1)
    min_col = max(min_col - padding, 0)
    max_col = min(max_col + padding, mask.shape[1] - 1)

    regions = [min_row, max_row, min_col, max_col]

    # Crop the region
    cropped_mask = mask[min_row : max_row + 1, min_col : max_col + 1]
    cropped_img = image[min_row : max_row + 1, min_col : max_col + 1]

    resized_mask = cv2.resize(cropped_mask, (1024, 1024), interpolation=cv2.INTER_AREA)
    resized_img = cv2.resize(cropped_img, (1024, 1024), interpolation=cv2.INTER_AREA)
    return (
        Image.fromarray(resized_img).convert("RGB"),
        Image.fromarray(resized_mask).convert("RGB"),
        regions,
        (cropped_img.shape[1], cropped_img.shape[0]),
    )


def fix_inpaint_prompt(adjective, facial_feature):
    match facial_feature:
        case "Hair":
            return f"{adjective} hair", None
        case "Eyebrows":
            return f"{adjective} eyebrows", None
        case "Eyes":
            return (
                f"{adjective} eyes",
                "deformed, bad, ugly, mutilated, mutation, disfigured, anime, drawing",
            )
        case "Nose":
            if adjective == "piercing":
                return f"nose {adjective}", None
            elif adjective == "no accessories":
                return "nose", "ring, piercing"
            else:
                print(f"Unknown adjective: {adjective} for feature: {facial_feature}")
                return None, None
        case "Ear":
            if adjective == "piercing":
                return f"{facial_feature} piercing", None
            elif adjective == "earring":
                return adjective, None
            elif adjective == "no accessories":
                return "ear", "jewelry, earring, piercing"
        case "Lips":
            if adjective == "no lipstick":
                return "closed human mouth", "lipstick"
            return f"{adjective} lips", None
        case "Beard":
            if adjective == "no facial hair":
                return (
                    "chin",
                    "beard, mustache, facial hair, hair, mouth, lips, lip, deformed, bad, ugly, mutilated, mutation, disfigured, anime, drawing",
                )
            elif adjective == "shaven":
                return "shaven", None
            elif adjective == "beard":
                return (
                    "beard",
                    "mustache, mouth, lips, lip, deformed, bad, ugly, mutilated, mutation, disfigured, anime, drawing",
                )
            elif adjective == "mustache":
                return "mustache", "beard"
            elif adjective == "beard and mustache":
                return "beard, mustache", None
        case "Face":
            return f"{adjective} person", NEGATIVE_PROMPT
