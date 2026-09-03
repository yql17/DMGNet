# DMGNet
A Decoupled Dual-stream Multi-scale Graph Network for Pathology-Genomics Cancer Survival Prediction


## Overview

This repository is the official implementation of **DMGNet**, a decoupled dual-stream multi-scale graph network for pathology-genomics cancer survival prediction. DMGNet integrates whole-slide images (WSIs) and transcriptomic data to model prognostically relevant pathological and molecular information.

The framework mainly includes:

* A genomic graph branch for modeling transcriptomic relationships among patients.
* A WSI branch for extracting multi-scale tissue-structure representations from pathological images.
* A risk-level fusion module for multimodal survival prediction.
* A  modality decoupling strategy (MDS) to alleviate optimization interference between modalities.

## Data Sources and Preprocessing
### Datasets
We use transcriptomic profiles and histopathological whole‑slide images (WSIs) from [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov/), covering three cancer cohorts:
- BLCA (bladder urothelial carcinoma): n=422
- LUAD (lung adenocarcinoma): n=574
- UCEC (uterine corpus endometrioid carcinoma): n=568

### Transcriptomic Data Preprocessing
1. Raw RNA‑seq read‑count matrices are downloaded from TCGA‑GDC portal.
2. TMM normalization is applied to eliminate compositional biases across samples.
3. Variance‑based gene filtering: remove zero‑variance genes; retain the **top 3000 highest‑variance genes**.
4. The filtered gene‑expression matrix is used to construct patient genomic graphs for subsequent model training.

### WSI Data Preprocessing
1. Multi‑threshold Otsu segmentation is performed to detect tissue regions and remove background artifacts.
2. Tissue regions are cropped into non‑overlapping **256 × 256 pixel patches** under **16× magnification**.
3. These patches serve as basic units for tissue feature extraction and pathological graph construction.

## Implementation & Training Details
- **Software Environment**: Python 3.8.13, PyTorch 2.4.1, CUDA 11.8
- **Hardware**: NVIDIA A100‑SXM4‑40GB GPU workstation
- **Optimizer**: Adam
- **Learning rate**:10^{-4}
- **Weight decay**: 10^{-5}
- **Training epochs**: 50
- **Batch size**: 6

## Code and Data‑Processing Availability
This repository provides model framework source code, patient filtering criteria, multi‑modal data preprocessing pipelines, feature‑extraction scripts, and experimental configuration files required to reproduce the main findings. All raw pathological and genomic data can be acquired from the public [TCGA database](https://portal.gdc.cancer.gov/).
No confidential or restricted datasets were used in the core experiments of this work.

## Reproducibility
1. Raw multi‑modal data: TCGA‑LUAD, TCGA‑BLCA, TCGA‑UCEC
2. Patient inclusion and exclusion rules and preprocessing workflows are provided within this repository.
3. Users can reproduce the primary results by downloading public TCGA resources and running the released preprocessing and model‑related code.

## Citation

If you find this work useful, please cite our paper after it is officially published.

```bibtex
@article{DMGNet,
  title={DMGNet: A Decoupled Dual-stream Multi-scale Graph Network for Pathology-Genomics Cancer Survival Prediction},
  author={To be updated},
  journal={To be updated},
  year={To be updated}
}
```

## Contact

For questions or further information, please contact the authors after the code is released.
