import os

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image


def prepare_image(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch


@torch.no_grad()
def inference(image, weights, model, device):
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
    else:
        raise ValueError(f"Weights not found from given path ({weights})")

    model.eval()

    resized_image = image.resize((512, 512), resample=Image.BILINEAR)
    transformed_image = prepare_image(resized_image)
    image_batch = transformed_image.to(device)

    output = model(image_batch)[
        0
    ]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only

    predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

    del output  # remove output from model

    predicted_mask = predicted_mask.astype("uint8")
    predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)

    return predicted_mask
