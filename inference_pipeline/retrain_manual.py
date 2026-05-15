"""
Quick retrain with manual params, bypassing Optuna.
Run from inference_pipeline/:
    python retrain_manual.py
"""

import pathlib
import numpy as np
import pandas as pd
import joblib
import config

PARAMS = {
    'num_leaves':        150,
    'max_depth':         11,
    'learning_rate':     0.02,
    'n_estimators':      1500,
    'min_child_samples': 100,
    'subsample':         0.5,
    'subsample_freq':    1,
    'colsample_bytree':  0.92,
    'reg_alpha':         5.0,
    'reg_lambda':        5.0,
    'min_split_gain':    0.0,
}

def main():
    # Load data
    features_file = (config.FEATURES_DIR / 'features.csv').resolve()
    df = pd.read_csv(features_file).dropna(subset=config.FEATURE_COLUMNS)
    train_trees = sorted(df['tree'].unique())[:config.N_TRAIN_TREES]
    df = df[df['tree'].isin(train_trees)]

    X = np.asarray(df[config.FEATURE_COLUMNS].values)
    y = np.asarray(df[['true_alpha', 'true_rho']].values)

    print(f"Training on {len(df)} samples, {X.shape[1]} features")
    print(f"Params: {PARAMS}")

    from models.lightgbm_model import LightGBMModel
    model = LightGBMModel(random_state=config.MASTER_SEED)
    model.set_params(**PARAMS)
    model.train(X, y)

    models_dir = pathlib.Path("models")
    out = models_dir / "lightgbm_model.pkl"
    joblib.dump(model, out)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
