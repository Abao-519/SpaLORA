

# SpaLORA: A Graph Neural Network Framework for Spatial Multi-Omics Integration Emphasizing Low-Expression Gene Signals

## Overview

SpaLORA is a graph neural network framework designed for integrating spatial multi-omics data with special emphasis on low-expression gene signals. The core innovation is a novel weighted reconstruction loss function that prioritizes low-expression genes while preserving the representation of highly expressed features, enabling more accurate spatial domain delineation and rare cell population identification.

SpaLORA incorporates a dual-attention mechanism that adaptively balances contributions from diverse modalities and spatial relationships. This is complemented by a weighted reconstruction loss that dynamically assigns higher weights to low-expression genes, which often provide critical information for identifying rare cell populations and defining spatial domain boundaries. By preserving non-dimensionally reduced transcriptomic features during processing, SpaLORA captures subtle, biologically significant signals that conventional methods typically miss. These innovations result in superior performance in clustering accuracy and fine-grained spatial domain identification when benchmarked against other advanced methods. Additionally, SpaLORA demonstrates exceptional robustness to common data quality issues such as dropout events and measurement noise, maintaining high performance even under severe perturbations.

## Installation

Create and set up a conda environment with SpaLORA:

```bash
# Create a new conda environment
conda create -n SpaLORA python=3.8

# Activate the environment
conda activate SpaLORA

# Install SpaLORA
pip install SpaLORA

# Install jupyter kernel for notebooks
pip install ipykernel

# Register the kernel
python -m ipykernel install --user --name=SpaLORA
```

## Usage

After setting up the environment, you can explore the tutorial notebooks in the `Tutorial` directory. Each notebook provides step-by-step guidance for analyzing different spatial multi-omics datasets:

- Human lymph node dataset
- P22 mouse brain coronal section dataset
- Human placenta architecture dataset


## Datasets

The preprocessed datasets and ground truth labels used in the paper can be downloaded from:
[https://drive.google.com/drive/folders/1sPUVnga4LcC5h2lxkskzUPKETNoCelmh?usp=drive_link](https://drive.google.com/drive/folders/1sPUVnga4LcC5h2lxkskzUPKETNoCelmh?usp=drive_link)

## Acknowledgments

The code for SpaLORA was developed based on the framework of [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue). We gratefully acknowledge their contribution to the spatial multi-omics research community.