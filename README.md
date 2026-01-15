# Diff-FIT

This framework was tested on Python 3.10.

## Dif-FIT Setup

For dependency installation you need Poetry. You need `Cmake` since this repository uses `dlib`.   
Make sure you have access to a GPU with at least 20GB VRAM and have at least 30GB free storage.

1. Run `poetry install`
2. Run `poetry run download_weights`
3. Run `poetry run diff_fit`

## Generate Datasets

With the following scripts you can generate datasets as described in the paper.   
To change the default parameters please navigate to [config](/generate_images/config.py).

To generate:
-  face images with random ids, run: `poetry run generate_images`.
-  inpainting images, run: `poetry run generate_inpainting_images`.
-   similar ids to existing ones (img2img), run: `poetry run generate_img2img_images`.

