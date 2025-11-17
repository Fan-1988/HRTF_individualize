# HRTF_individualize
This repository contains two end-to-end pipelines for individualizing Head-Related Transfer Functions (HRTFs) using different types of input data.
Both pipelines include data processing, model training, and inference code.

## What is HRTF?

A Head-Related Transfer Function (HRTF) describes how sound from a given direction is filtered by a listener’s head, torso, and outer ear (pinna) before reaching the ear drum.
HRTFs are essential for:3D/Spatial Audio，Virtual/Augmented Reality，Audio rendering in games/films，Hearing-aid and audio device personalization

Because everyone has a unique anatomical shape, HRTFs vary between individuals. This motivates HRTF individualization, where we predict personalized HRTFs from anatomical data.

This Repository provide two separate models for individualized HRTF prediction:
Model A — Based on 3D Ear Geometry
Model B — Based on Anthropometric Features (head, torso, ear parameters)

Each model contains a complete workflow:
✔ Data processing
✔ Model architecture
✔ Training pipeline
✔ Inference scripts
✔ Pretrained models

## Model A: 3D Ear Model → HRTF (ResNet + Multi-Task Learning)
This model predicts individualized HRTFs from a 3D ear mesh.
We process the 3D geometry and feed it into a 3D ResNet backbone with a multi-task learning structure.

Input: 3D ear model (cropped, normalized)
Architecture: 3D ResNet
Learning Strategy: Multi-task
Task 1: Predict frequency-domain HRTFs
Task 2: Predict spherical harmonic (SH) representation
Loss: LSD

## Model B: Anthropometric Parameters → HRTF (U-Net)
This model uses anthropometric measurements, including:
Head size
Torso dimension
Ear height/width
Concha-related parameters
Shoulder parameters

We map these features to HRTFs using a U-Net architecture.
Output: HRTF at all spatial sampling points


Suitable when 3D geometry is unavailable
