---
title: "CostalSeg Coastal Erosion Platform"
excerpt: "UW capstone system for coastal erosion monitoring with DeepLabV3+ segmentation, ViT-H-14 quality control, and SuperGlue alignment<br/><img src='/images/costalseg-overlay.webp'>"
collection: portfolio
---

## Overview
CostalSeg is a University of Washington capstone project that builds an end-to-end workflow for community-driven coastal erosion monitoring. The toolkit ingests user photos, filters low-quality submissions, aligns multi-angle imagery, and produces seven-class shoreline segmentations powered by a DeepLabV3+ model with an EfficientNet-B6 backbone. The production checkpoints reach a 0.93 IoU on the program's labeled datasets and integrate directly with the Applied Physics Laboratory's MyCoast submission flow.

## My Contributions
- Led model training and evaluation for the Metal Marcy and Silhouette Jaenette beaches, including data preparation scripts and packaged pretrained checkpoints.
- Implemented the Gradio-based browser app (`app.py`) that wraps inference, overlay rendering, and batch processing for field researchers.
- Authored documentation for model retraining, outlier detection, and deployment to Hugging Face Spaces.

## Highlights
- **Multi-class segmentation pipeline** - DeepLabV3Plus + EfficientNet-B6 models exported through `segmentation_models_pytorch`; utilities for both scratch training and loading shared checkpoints.
- **Interactive analyst UI** - One-click Gradio experience with layered visualization, downloadable masks, and batch folder processing.
- **Outlier detection guardrail** - ViT-H-14 feature embeddings compare incoming photos against curated references to keep the training corpus clean.
- **Perspective correction** - SuperGlue keypoint matching plus homography warping keeps shoreline contours consistent across community uploads.
- **Performance tooling** - `model_performance_test.py` records inference throughput, GPU utilization, and FLOP efficiency across batch sizes and image resolutions.
- **Hosted demos** - Three Hugging Face Spaces (MetalMarcy, SilhouetteJaenette, CostalSegment) expose the workflow to project partners without local setup.

## Technical Stack
`Python`, `PyTorch`, `segmentation_models_pytorch`, `Gradio`, `Hugging Face transformers`, `OpenCV`, `NumPy`

## Learn More
- GitHub repository: [cxh42/CostalSeg](https://github.com/cxh42/CostalSeg)
- Hugging Face demo: [AveMujica/CostalSegment](https://huggingface.co/spaces/AveMujica/CostalSegment)
