# Diff-FIT

Diff-FIT is an open-source framework for realistic facial composite generation using diffusion models. It supports both full face generation and targeted edits through an interactive workflow that integrates face-segmentation–guided inpainting for precise region editing and landmark-assisted Lightning Drag for fast, high-quality, controllable results.


## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Generation](#dataset-generation)
- [Components](#components)
- [Metrics](#metrics)

## Prerequisites

- Python 3.10
- Poetry (for dependency management)
- CMake (required for `dlib`)
- GPU with at least 20GB VRAM
- At least 30GB free storage

## Installation

1. Install dependencies:  
   ```bash
   poetry install
   ```

2. Download weights:  
   ```bash
   poetry run download_weights
   ```

## Usage

Run the Diff-FIT application:  
```bash
poetry run diff_fit
```

Once the application loads, navigate to `http://localhost:7860` in your web browser to access the interface.

## Dataset Generation

With the following scripts, you can generate datasets as described in the paper. To change the default parameters, please navigate to [config](/generate_images/config.py).

| Dataset Type | Command |
|--------------|---------|
| Face images with random IDs | `poetry run generate_images` |
| Inpainting images | `poetry run generate_inpainting_images` |
| Similar IDs to existing ones (img2img) | `poetry run generate_img2img_images` |

## Components

Diff-FIT integrates the following components for enhanced functionality:

- **[Lightning Drag](https://github.com/magic-research/LightningDrag)**: An interactive tool for precise face editing, integrated with Diff-FIT.
- **[Face Parsing](https://github.com/yakhyo/face-parsing)**: A face parsing model for semantic segmentation of facial features.
- **[RealVis XL V5.0 Lightning](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning)**: The underlying SDXL Lightning model for high-quality image generation, developed by SG161222.

## Metrics

This repository provides scripts for generating datasets as described in the paper. To reproduce and evaluate the results, you can use the following metric repositories and datasets:

- **[CMMD](https://github.com/google-research/google-research/tree/master/cmmd)**: A metric for evaluating image quality between synthetic and real datasets.
- **[eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA)**: A metric for image quality assessment using diffusion models.
- **[VQAScore](https://github.com/linzhiqiu/t2v_metrics)**: Visual Quality Answering metric for evaluating prompt adherence.
- **[IRS](https://github.com/MischaD/BeyondFID)**: Image Retrieval Score for evaluating dataset diversity.
- **[Cosine Similarity with ArcFace](https://learnopencv.com/face-recognition-with-arcface/)**: A metric for face recognition and similarity evaluation using ArcFace embeddings.
- **[FFHQ](https://github.com/NVlabs/ffhq-dataset)**: A high-quality dataset of human faces for training and evaluation.

## Citation

If you use the code or results from this repository, please cite the ID-Booth paper:

```
To Be Added.
}
```

## Acknowledgements

Supported in parts by the Slovenian Research and Innovation Agency (ARIS) through the Research Programmes P2-0250 (B) "Metrology and Biometric Systems" and P2-0214 (A) “Computer Vision”, the ARIS Project J2-50065 "DeepFake DAD" and the ARIS Young Researcher Programme.

