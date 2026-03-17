# FloodGraphFlow-XGB

Solution for the Kaggle competition 

[**UrbanFloodBench: Flood Modelling**](https://www.kaggle.com/competitions/urban-flood-modelling/overview)

<img width="400" height="233" alt="Gemini_Generated_Image_vr2b04vr2b04vr2b" src="https://github.com/user-attachments/assets/c47540cb-5d99-4e67-86e4-10e874e2c8fc" />

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
2. Place the extracted folders inside `./`

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

The models for Cities 1 and 2 are trained with the following commands:
```
# Train Model_1
python scripts/run_floodgraphflow_xgb.py \
    --config configs/model1_best.yaml \
    --backend xgboost_gpu \
    --save_model_path saved_models/model1_best.pkl 

# Train Model_2
python scripts/run_floodgraphflow_xgb.py \
    --config configs/model2_best.yaml \
    --backend xgboost_gpu \
    --save_model_path saved_models/model2_best.pkl 
```

## Inference
Model inference can be done with the following commands:
```
# Model_1 test predictions
python scripts/run_floodgraphflow_xgb.py \
    --config configs/model1_best.yaml \
    --backend xgboost_cpu \
    --load_model_path saved_models/model1_best.pkl \
    --dump_test_predictions predictions/model1_test_predictions.parquet

# Model_2 test predictions
python scripts/run_floodgraphflow_xgb.py \
    --config configs/model2_best.yaml \
    --backend xgboost_cpu \
    --load_model_path saved_models/model2_best.pkl \
    --dump_test_predictions predictions/model2_test_predictions.parquet
```
## Approach

We use a **graph-aware stacked XGBoost pipeline** rather than a single end-to-end sequence model.

Flood behavior in this task is **not purely local**: each node depends on *upstream inflow*, *downstream blockage*, *storage*, and *boundary conditions*. A single flat regressor struggled to represent these different hydraulic regimes.

By combining **graph-derived features**, **physics-inspired hydraulic proxy features**, **auxiliary flow surrogates (`qnet`, `qin`, `qout`)**, and a **two-stage regime-aware predictor**, the model captures propagation, retention, and delayed drainage more reliably than local rainfall and water-level features alone.

---

### Model Overview

At a high level, the model works in **four stages**.

**Stage 1 — Graph-aware feature construction**

> We build node-level temporal features from rainfall, water level, and static network attributes, then augment them with **graph features**, **boundary indicators**, and **mass-deficit storage proxies** that encode coarse hydraulic structure.

**Stage 2 — Auxiliary flow prediction**

> We train **Stage-A models** to predict latent hydraulic quantities such as **net flow**, **inflow**, and **outflow**. Their **out-of-fold predictions** are fed back into the feature set.

**Stage 3 — Regime-aware prediction**

> A **two-stage XGBoost predictor** combines **regime classification** with **conditional regression**, allowing the model to treat calm, rising, and storage-dominated states differently.

**Stage 4 — Final submission assembly**

> The full pipeline is trained separately for **`Model_1`** and **`Model_2`**, and their predictions are merged into the final submission.

---
### Preprocessing

---

### Hyperparameters

---

### Feature Engineering

The final submission used a total of 250+ derived features (32 stacked stage A, 213 stage B). A detailed list of all included features is included in `FEATURES.md`.

The strongest feature families in the final branch were:

#### 1. Graph-propagated water-level context

> These features summarize nearby hydraulic state over the drainage graph rather than using only the local node.

**`fe_graph_pulse`**
  - Aggregates warm-start water level over upstream neighbors
  - Produces upstream level sum and mean

**`fe_graph_hop2_features`**
  - Extends this idea to 2-hop upstream context
  - Produces 2-hop upstream sum and mean

**`fe_level_imbalance_features`**
  - Measures local hydraulic imbalance using upstream and downstream support
  - Includes:
    - `level_imbalance = upstream_sum - downstream_sum`
    - `level_gradient = local_wl - upstream_mean`
    - `downstream_pull = downstream_mean - local_wl`

#### 2. Stage-A net-flow stack (`qnet`)

> We train an auxiliary model to predict net flow, then feed those predictions back into the main model.

**`qnet_stack`**
  - Predicts latent net flow (`qhat`)
  - Can expose raw `qhat`, rolling summaries, cumulative values, and lags

**`qnet_phys_baseline_feature`**
  - Adds a physics-style baseline:
  - `predicted_water_level_baseline = y_base + qhat`

**`fe_qhat_graph2`**
  - Pushes predicted `qhat` through the graph
  - Builds upstream sum, upstream mean, and upstream-downstream imbalance features

**`fe_qhat_graph2_hop2`**
  - Extends graph summaries of `qhat` to 2-hop upstream structure

#### 3. Stage-A inflow / outflow stack (`qin`, `qout`)

> These features expose directional transport structure that is hard to recover from local predictors alone.

- **`qinout_stack`**
  - Trains separate auxiliary models for inflow and outflow
  - Appends predicted `qin` and `qout`
  - Includes cumulative, rolling, and lagged summaries

#### 4. Basin mass-deficit and storage proxy features

> These are the main mass-balance-like engineered features.

- **`fe_basin_mass_deficit_features`**
  - Computes current basin storage from denormalized node `water_volume`
  - Computes accumulated rainfall volume since warm start using node area
  - Defines:
    - `expected = warm_start_basin_storage + accumulated_rain_volume`
    - `basin_mass_deficit = expected - current_basin_storage`

---

### Model Ablations
Model | Addition | City 2 Score (local validation)
A | Baseline XGBoost |
B | A + graph features |
C | B + 
D | C + mass deficit features
