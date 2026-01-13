import numpy as np
from PIL import Image

import diff_fit.transformation_utils.utils as utils
from diff_fit import LIGHTNING_DRAG_WEIGHTS
from diff_fit.transformation_utils.constants import LANDMARK_INDICES
from lightning_drag.drag_utils.ui_utils import LightningDragUI


class FaceTransformation:
    def __init__(self):
        print("\nInitializing LightningDrag...\n")
        # self.lightning_drag = LightningDragUI(
        #     f"{LIGHTNING_DRAG_WEIGHTS}/dreamshaper-8-inpainting/",
        #     vae_path=f"{LIGHTNING_DRAG_WEIGHTS}/sd-vae-ft-mse/",
        #     ip_adapter_path=f"{LIGHTNING_DRAG_WEIGHTS}/IP-Adapter/models/",
        #     lightning_drag_path=f"{LIGHTNING_DRAG_WEIGHTS}/lightning-drag-sd15",
        #     lcm_lora_path=f"{LIGHTNING_DRAG_WEIGHTS}/lcm-lora-sdv1-5",
        # )
        self.image = None
        self.stored_latents = None
        self.latents = None
        self.landmarks = None

    def init_image(self, image, index=None, detect_landmarks=True):
        if index is None:
            image = np.array(image.convert("RGB"))
        else:
            image = np.array(image[index][0].convert("RGB"))
        self.image = image
        if detect_landmarks:
            self.landmarks = utils.detect_landmarks(self.image)
        self.latents = None
        self.stored_latents = None
        return Image.fromarray(self.image).convert("RGB"), Image.fromarray(
            self.image
        ).convert("RGB")

    def change_output(self, image):
        self.image = np.array(image[1])
        self.landmarks = utils.detect_landmarks(self.image)
        self.latents = self.stored_latents
        return image[1], image[1]

    def execute_transform(self, points, mask=None):
        if len(points) == 0 or len(points) % 2 != 0:
            return [
                Image.fromarray(self.image).convert("RGB"),
                Image.fromarray(self.image).convert("RGB"),
            ], Image.fromarray(self.image).convert("RGB")

        if mask is None:
            ###
            points = (
                [(x * 2, y * 2) for x, y in points]
                if self.image.shape[0] > 512
                else points
            )
            mask = utils.get_mask(points, size=self.image.shape[0])
        else:
            mask = mask.convert("RGB")
            mask = np.array(mask)
            mask[mask == 255] = 1

        image, latents = self.lightning_drag.run_drag(
            seed=0,
            source_image=self.image,
            mask=mask,
            points=points,
            num_inference_steps=8,
            guidance_scale_points=3,
            guidance_scale_decay="inv_square",
            latents=self.latents,
        )
        self.stored_latents = latents
        transformation_image = Image.fromarray(
            utils.get_transformation_image(self.image, mask, points)
        ).convert("RGB")
        return (
            [
                Image.fromarray(self.image).convert("RGB"),
                Image.fromarray(image).convert("RGB"),
            ],
            transformation_image,
            "LightningDrag",
            None,
            None,
            0,
            None,
            None,
            8,
            3,
            None,
            points,
            mask,
            transformation_image,
            Image.fromarray(image).convert("RGB"),
        )

    def transform(
        self,
        jawline_vertical_offset: int = 0,
        jawline_horizontal_offset: int = 0,
        forehead_vertical_offset: int = 0,
        forehead_horizontal_offset: int = 0,
        brow_vertical_offset: int = 0,
        brow_horizontal_offset: int = 0,
        eye_vertical_offset: int = 0,
        eye_horizontal_offset: int = 0,
        eye_size_offset: int = 0,
        eye_rotation_offset: int = 0,
        nose_vertical_offset: int = 0,
        nose_horizontal_offset: int = 0,
        nose_width_offset: int = 0,
        lips_vertical_offset: int = 0,
        lips_horizontal_offset: int = 0,
        lips_volume_offset: int = 0,
    ):
        points = []

        if jawline_vertical_offset != 0 or jawline_horizontal_offset != 0:
            points += self.transform_jawline(
                jawline_vertical_offset, jawline_horizontal_offset
            )
        if forehead_vertical_offset != 0 or forehead_horizontal_offset != 0:
            points += self.transform_forehead(
                forehead_vertical_offset, forehead_horizontal_offset
            )

        if brow_vertical_offset != 0 or brow_horizontal_offset != 0:
            points += self.transform_brow(brow_vertical_offset, brow_horizontal_offset)

        if (
            eye_vertical_offset != 0
            or eye_horizontal_offset != 0
            or eye_size_offset != 0
            or eye_rotation_offset != 0
        ):
            points += self.transform_eye(
                eye_vertical_offset,
                eye_horizontal_offset,
                eye_size_offset,
                eye_rotation_offset,
            )

        if (
            nose_vertical_offset != 0
            or nose_horizontal_offset != 0
            or nose_width_offset != 0
        ):
            points += self.transform_nose(
                nose_vertical_offset, nose_horizontal_offset, nose_width_offset
            )

        if (
            lips_vertical_offset != 0
            or lips_horizontal_offset != 0
            or lips_volume_offset != 0
        ):
            points += self.transform_lips(
                lips_vertical_offset, lips_horizontal_offset, lips_volume_offset
            )

        return self.execute_transform(points)

    def transform_lips(
        self, vertical_offset: int, horizontal_offset: int, volume_offset: int
    ):
        if volume_offset == 0:
            selected_points = utils.get_selected_points(
                self.landmarks, selected_landmarks=LANDMARK_INDICES["lips"]
            )

            target_points = utils.move_y(selected_points, vertical_offset)
            target_points = utils.move_x(target_points, horizontal_offset)

        else:
            ## only for lip volume
            selected_points = utils.get_selected_points(
                self.landmarks,
                selected_landmarks=[49, 50, 51, 52, 53, 55, 56, 57, 58, 59],
            )
            target_points = utils.move_y(selected_points[:5], -volume_offset)
            target_points += utils.move_y(selected_points[5:], volume_offset)

        points = utils.prepare_points(selected_points, target_points)
        points = utils.clamp_points(points)

        return points

    def transform_brow(self, vertical_offset: int, horizontal_offset: int):
        left_brow = utils.get_selected_points(
            self.landmarks, selected_landmarks=LANDMARK_INDICES["left_brow"]
        )

        target_points = utils.move_y(left_brow, vertical_offset)
        target_points = utils.move_x(target_points, horizontal_offset)
        points = utils.prepare_points(left_brow, target_points)

        right_brow = utils.get_selected_points(
            self.landmarks, selected_landmarks=LANDMARK_INDICES["right_brow"]
        )

        target_points = utils.move_y(right_brow, vertical_offset)
        target_points = utils.move_x(target_points, -horizontal_offset)
        points += utils.prepare_points(right_brow, target_points)

        points = utils.clamp_points(points)

        return points

    def transform_nose(
        self, vertical_offset: int, horizontal_offset: int, width_offset: int
    ):
        selected_points = utils.get_selected_points(
            self.landmarks, selected_landmarks=LANDMARK_INDICES["nose"]
        )
        if width_offset == 0:
            target_points = utils.move_y(selected_points, vertical_offset)
            target_points = utils.move_x(target_points, horizontal_offset)

        else:
            selected_points = (
                utils.move_x(selected_points[:4], -8)  # 4
                + selected_points[4:6]  # 2
                + utils.move_x(selected_points[:4], 8)  # 4
                + selected_points[7:9]  # 2
            )
            target_points = utils.move_x(
                selected_points[:4] + selected_points[4:6], -width_offset
            )
            target_points += utils.move_x(
                selected_points[6:10] + selected_points[10:], width_offset
            )

        points = utils.prepare_points(selected_points, target_points)
        points = utils.clamp_points(points)

        return points

    def transform_jawline(self, vertical_offset: int, horizontal_offset: int):
        selected_points = utils.get_selected_points(
            self.landmarks, selected_landmarks=LANDMARK_INDICES["jaw"]
        )
        target_points = utils.move_y(selected_points, vertical_offset)
        target_points = (
            utils.move_x(target_points[:4], -horizontal_offset) + target_points[4:]
        )
        target_points = target_points[:5] + utils.move_x(
            target_points[5:], horizontal_offset
        )
        if vertical_offset == 0:
            selected_points.pop(4)
            target_points.pop(4)
        points = utils.prepare_points(selected_points, target_points)
        points = utils.clamp_points(points)

        return points

    def transform_forehead(self, vertical_offset: int, horizontal_offset: int):
        selected_points = utils.get_selected_points(
            self.landmarks, selected_landmarks=LANDMARK_INDICES["jaw"]
        )
        selected_points = utils.mirror_y(
            selected_points,
            utils.get_selected_points(
                self.landmarks, selected_landmarks=LANDMARK_INDICES["center_face"]
            )[0],
        )
        target_points = utils.move_y(selected_points, -vertical_offset)
        target_points = (
            utils.move_x(target_points[:4], -horizontal_offset) + target_points[4:]
        )
        target_points = target_points[:5] + utils.move_x(
            target_points[5:], horizontal_offset
        )
        if vertical_offset == 0:
            selected_points.pop(4)
            target_points.pop(4)
        points = utils.prepare_points(selected_points, target_points)
        points = utils.clamp_points(points)

        return points

    def transform_eye(
        self,
        vertical_offset: int,
        horizontal_offset: int,
        size_offset: int,
        eye_rotation_offset: int,
    ):
        selected_points = utils.get_selected_points(
            self.landmarks, selected_landmarks=LANDMARK_INDICES["left_eye"]
        )

        center, width, height, phi = utils.fit_elipse(selected_points)
        x, y = center
        left_eye = utils.get_ellipse_linspace(x, y, width, height, phi)
        left_eye_offset = utils.get_ellipse_linspace(
            x - horizontal_offset,
            y + vertical_offset,
            width + size_offset,
            height + size_offset,
            phi + np.deg2rad(eye_rotation_offset),
        )

        points = utils.prepare_points(
            list(zip(left_eye[0], left_eye[1])),
            list(zip(left_eye_offset[0], left_eye_offset[1])),
        )

        selected_points = utils.get_selected_points(
            self.landmarks, selected_landmarks=LANDMARK_INDICES["right_eye"]
        )

        center, width, height, phi = utils.fit_elipse(selected_points)
        x, y = center
        right_eye = utils.get_ellipse_linspace(x, y, width, height, phi)
        right_eye_offset = utils.get_ellipse_linspace(
            x + horizontal_offset,
            y + vertical_offset,
            width + size_offset,
            height + size_offset,
            phi - np.deg2rad(eye_rotation_offset),
        )

        points += utils.prepare_points(
            list(zip(right_eye[0], right_eye[1])),
            list(zip(right_eye_offset[0], right_eye_offset[1])),
        )

        return points
