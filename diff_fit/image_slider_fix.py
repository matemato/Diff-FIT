from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Tuple, Union

import numpy as np
from gradio import processing_utils, utils  # type: ignore
from gradio.data_classes import FileData, GradioRootModel  # type: ignore
from PIL import Image as _Image  # using _ to minimize namespace pollution

_Image.init()  # fixes https://github.com/gradio-app/gradio/issues/2843


class SliderData(GradioRootModel):
    root: Union[Tuple[FileData | None, FileData | None], None]


image_variants = _Image.Image | np.ndarray | str | Path | FileData | None
image_tuple = (
    tuple[str, str]
    | tuple[_Image.Image, _Image.Image]
    | tuple[np.ndarray, np.ndarray]
    | None
)


def _postprocess_image(self, y: image_variants):
    if isinstance(y, np.ndarray):
        path = processing_utils.save_img_array_to_cache(
            y, cache_dir=self.GRADIO_CACHE, format="png"
        )
    elif isinstance(y, _Image.Image):
        path = processing_utils.save_pil_to_cache(
            y, cache_dir=self.GRADIO_CACHE, format="png"
        )
    elif isinstance(y, (str, Path)):
        path = y if isinstance(y, str) else str(utils.abspath(y))
    else:
        raise ValueError("Cannot process this value as an Image")

    return path
