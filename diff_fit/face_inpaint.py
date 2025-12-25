import cv2
import numpy as np
import torch
from PIL import Image

from diff_fit import RESNET_WEIGHTS
from diff_fit.inpaint_utils.constants import MODEL, NUM_CLASSES, SEGMENTATION_CLASSES
from face_parsing.inference import inference
from face_parsing.segmentation_models.bisenet import BiSeNet


class FaceInpaint:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = BiSeNet(NUM_CLASSES, backbone_name=MODEL).to(self.device)
        self.mask = None

    def init_inpaint(self, image):
        self.mask = inference(
            image=image,
            weights=RESNET_WEIGHTS,
            model=self.model,
            device=self.device,
        )

    def show_mask(self, image, face_attribute: str):
        face_classes = SEGMENTATION_CLASSES[face_attribute]
        mask = np.copy(self.mask)
        for face_class in face_classes:
            mask[mask == face_class] = 255
        mask[mask != 255] = 0

        # if face_attribute != "Eyes":
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=3)

        ### if face_atrtribute is beard, nose or ear use only lower half
        if face_attribute in ["Beard", "Nose", "Ear"]:
            rows = np.where(mask == 255)[0]
            mask[rows.min() : int(rows.min() + (rows.max() - rows.min()) // 1.7), :] = 0

        if face_attribute == "Beard":
            for face_class in (
                SEGMENTATION_CLASSES["Lips"] + SEGMENTATION_CLASSES["Nose"]
            ):
                mask = np.where(self.mask != face_class, mask, 0)

        mask = cv2.resize(mask, image.size)

        blended_image = cv2.addWeighted(np.array(image), 1, mask, 0.4, 0)

        return Image.fromarray(mask).convert("RGB"), Image.fromarray(
            blended_image
        ).convert("RGB")

    def remove_background(self, image):
        mask = np.copy(self.mask)
        mask[mask == 16] = 0
        mask[mask == 0] = 255
        mask[mask != 255] = 0

        mask = cv2.resize(mask, image.size)

        blended_image = cv2.addWeighted(np.array(image), 1, mask, 1, 0)
        return Image.fromarray(blended_image).convert("RGB")
