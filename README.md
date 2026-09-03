# DMGNet
A Decoupled Dual-stream Multi-scale Graph Network for Pathology-Genomics Cancer Survival Prediction


## Overview

This repository is the official implementation of **DMGNet**, a decoupled dual-stream multi-scale graph network for pathology-genomics cancer survival prediction. DMGNet integrates whole-slide images (WSIs) and transcriptomic data to model prognostically relevant pathological and molecular information.

The framework mainly includes:

* A genomic graph branch for modeling transcriptomic relationships among patients.
* A WSI branch for extracting multi-scale tissue-structure representations from pathological images.
* A risk-level fusion module for multimodal survival prediction.
* A  modality decoupling strategy (MDS) to alleviate optimization interference between modalities.

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
