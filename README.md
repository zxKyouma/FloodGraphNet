# FloodGraphFlow-XGB
Solution for the Kaggle competition 

[**UrbanFloodBench: Flood Modelling**](https://www.kaggle.com/competitions/urban-flood-modelling/overview)

---

A graph-aware XGBoost pipeline for urban flood prediction using hydraulic context features, flow proxies, and mass-balance approximations.

Developed by team **NSEOverflow**
- Brandon Low — [@zxKyouma](https://github.com/zxKyouma)
- Daphne Chu — [@D-Keii](https://github.com/D-Keii)

## Environment Setup

```bash
conda env create -f environment.yml
conda activate flood
```

## Dataset
Download the [UrbanFloodBench dataset](https://drive.google.com/file/d/18XT8NlOfWOnR64mi6faKOR2mVzqpaLSA/view)

After downloading:
1. Extract the archive
2. Place the extracted folders inside `/Models`

## Repository Structure
Your directory should look like this:
```
FloodGraphFlow-XGB
├── configs
├── scripts
├── utils
├── Models
│   ├── Model_1
│   │   ├── train
│   │   ├── test
│   │   └── csv_features_stats.yaml
│   └── Model_2
│       ├── train
│       ├── test
│       └── csv_features_stats.yaml
├── environment.yml
└── README.md
```

## Model Training

## Inference
