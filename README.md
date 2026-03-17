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

## Approach

We use a graph-aware stacked XGBoost pipeline rather than a single end-to-end sequence model. 

Flood behavior in this task is not purely local: each node depends on upstream inflow, downstream blockage, storage, and boundary conditions. A single flat regressor struggled to represent these different hydraulic regimes. By combining graph-derived features, physics- inspired hydraulic proxy features, auxiliary flow surrogates (`qnet`, `qin`, `qout`), and a two-stage regime-aware predictor, the model captures propagation, retention, and delayed drainage more reliably than local rainfall and water-level features alone.

At a high level, the model works in four stages. 

- First, we build node-level temporal features from rainfall, water level, and static network attributes, then augment them with graph features, boundary and endpoint indicators, and storage or mass-deficit proxy features that encode coarse hydraulic structure. 

- Second, we train auxiliary stage-A models to predict latent hydraulic quantities such as net flow, inflow, outflow, and selected event-timing targets; their out-of-fold predictions are fed back in as features. 

- Third, a two-stage XGBoost predictor combines regime classification with conditional regression to handle different hydraulic behaviors more effectively than a single global regressor. 

- Finally, the same pipeline is trained separately for `Model_1` and `Model_2`, and the two prediction files are merged into the final submission.

### Model Overview

- Graph-aware feature engineering over the drainage network
- Stacked predictions for latent hydraulic variables (`qnet`, `qin`, `qout`)
- Physics-inspired and mass-deficit features for storage and drainage imbalance
- Two-stage XGBoost prediction with regime classification plus conditional regression
- Separate training for `Model_1` and `Model_2`, followed by submission merging


