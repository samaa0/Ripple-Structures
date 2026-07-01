# ML, Computing, And Plotting

This folder contains the final model training, testing, fusion, data checking, and plotting scripts used by Ripple Structures.

## Layout

- `models/` contains the final public model/config artifacts.
- `data/` documents expected dataset schemas. Raw datasets are intentionally excluded.
- `ripple_paths.py` centralizes repo-relative model/data paths.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run scripts from this folder unless a script documents otherwise:

```sh
cd software/ml
python feature_test.py
```

Set `RIPPLE_ML_MODEL_DIR` to point at a different model-artifact directory and `RIPPLE_ML_DATA_DIR` to point at local private datasets.

## Published Model Artifacts

- `models/final_raw_data_branch_model.pt`
- `models/best_lgbm_model_rank_1.txt`
- `models/best_meta_learner_model.pkl`
- `models/feature_scaler.pkl`
- `models/best_fusion_params.json`
- `models/model_rank_1_performance.json`
- `models/model_rank_1_performance_MLP.json`
- `models/optuna_feature_data_branch_study.db`
- `models/raw_data_branch_gru_study.db`

Training CSVs, logs, and checkpoints are not committed.
