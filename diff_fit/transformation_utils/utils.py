import cv2
import dlib  # type: ignore
import gradio as gr  # type: ignore
import numpy as np
from ellipse import LsqEllipse  # type: ignore
from PIL import Image

from diff_fit import FACE_LANDMARKS_WEIGHTS
from diff_fit.transformation_utils.constants import MASK_OFFSET


def detect_landmarks(img):
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(FACE_LANDMARKS_WEIGHTS))
    faces = detector(grayscale_img)

    if len(faces) == 0:
        return None

    return predictor(grayscale_img, faces[0])


def fit_elipse(points):
    reg = LsqEllipse().fit(np.array(points))
    return reg.as_parameters()


def get_ellipse_linspace(x0, y0, width, height, phi, num_points=10):
    # Generate linspace for angle t from 0 to 2*pi
    t = np.linspace(0, 2 * np.pi, num_points)

    # Parametric equations of the ellipse before rotation
    x = width * np.cos(t)
    y = height * np.sin(t)

    # Apply the rotation matrix and translation to the center (x0, y0)
    x_rot = x0 + x * np.cos(phi) - y * np.sin(phi)
    y_rot = y0 + x * np.sin(phi) + y * np.cos(phi)

    x = [int(x_i) for x_i in x_rot][1:]
    y = [int(y_i) for y_i in y_rot][1:]

    return x, y


def get_selected_points(landmarks, selected_landmarks):
    return [[landmarks.part(i).x, landmarks.part(i).y] for i in selected_landmarks]


def move_y(points, offset):
    return [[point[0], point[1] + offset] for point in points]


def move_x(points, offset):
    return [[point[0] + offset, point[1]] for point in points]


def mirror_y(points, mirror_point):
    return [[x_i, int(y_i - 2.1 * (y_i - mirror_point[1]))] for x_i, y_i in points]


def prepare_points(selected_points, target_points):
    points = []
    for i in range(len(selected_points)):
        points.append(selected_points[i])
        points.append(target_points[i])
    return points


def clamp_points(points, min_val=0, max_val=511):
    clamped_points = []
    for x, y in points:
        clamped_x = max(min(x, max_val), min_val)
        clamped_y = max(min(y, max_val), min_val)
        clamped_points.append([clamped_x, clamped_y])
    return clamped_points


def dict_to_list(dict):
    list = []
    for value in dict.values():
        list += value
    return list


def get_mask(points, size=1024):
    mask = np.zeros((size, size, 3), dtype=np.uint8)
    min_x = max(0, min([points[0] for points in points]) - MASK_OFFSET)
    max_x = min(size - 1, max([points[0] for points in points]) + MASK_OFFSET)
    min_y = max(0, min([points[1] for points in points]) - MASK_OFFSET)
    max_y = min(size - 1, max([points[1] for points in points]) + MASK_OFFSET)
    mask[min_y:max_y, min_x:max_x, :] = 1
    return mask


def get_transformation_image(img, mask, points, save=False):
    image = img.copy()
    points = np.array(points)
    for i in range(0, len(points), 2):
        cv2.circle(image, points[i], 5, (255, 0, 0), -1)
        cv2.circle(image, points[i + 1], 5, (0, 0, 255), -1)
        cv2.arrowedLine(image, points[i], points[i + 1], (0, 0, 0), 2, tipLength=0.5)

    mask = mask * 255
    combined = cv2.addWeighted(image, 1, mask, 0.2, 0)

    if save:
        cv2.imwrite("drag_temp.png", cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

    return combined


def update_masked_image(inpaint_canvas):
    init_image, mask = inpaint_canvas["background"], inpaint_canvas["layers"][0]
    blended_image = cv2.addWeighted(np.array(init_image), 1, np.array(mask), 0.4, 0)
    return blended_image


def get_points(img, points, selected_point: gr.SelectData):
    points.append(selected_point.index)
    img = np.array(img)
    for idx, point in enumerate(points):
        if idx % 2 == 0:
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
            prev_point = point
        else:
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
            cv2.arrowedLine(img, prev_point, point, (0, 0, 0), 2, tipLength=0.5)

    return Image.fromarray(img).convert("RGB")


def remove_points():
    return []
