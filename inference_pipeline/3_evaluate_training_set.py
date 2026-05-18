# load in a subset of the traning set and return the MSE + R^2
import pathlib

import joblib

import config
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def load_training_data():
    """
    Load training data for the model from the precomputed features file.
    """
    features_file = (config.FEATURES_DIR / 'features.csv').resolve()
    merged_df = pd.read_csv(features_file)
    merged_df = merged_df.dropna(subset=config.FEATURE_COLUMNS)

    train_trees = sorted(merged_df['tree'].unique())[:config.N_TRAIN_TREES]
    train_df = merged_df[merged_df['tree'].isin(train_trees)]

    X_train = train_df[config.FEATURE_COLUMNS].values
    y_train = train_df[['true_alpha', 'true_rho']].values
    train_groups = train_df['tree'].values

    return X_train, y_train, train_groups


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse_alpha = mean_squared_error(y[:, 0], y_pred[:, 0])
    mse_rho = mean_squared_error(y[:, 1], y_pred[:, 1])
    r2_alpha = r2_score(y[:, 0], y_pred[:, 0])
    r2_rho = r2_score(y[:, 1], y_pred[:, 1])

    return {
        'mse_alpha': mse_alpha,
        'mse_rho': mse_rho,
        'r2_alpha': r2_alpha,
        'r2_rho': r2_rho
    }

def main():
    print("=" * 60)
    print("EVALUATING TRAINING SET PERFORMANCE")
    print("=" * 60)
    models_dir = pathlib.Path("models")

    X_train, y_train, _ = load_training_data()

    # load best neural net model
    model_file = models_dir / "lightgbm_model.pkl"
    model = joblib.load(model_file)
    print(f"Loaded model from: {model_file}")

    results = evaluate_model(model, X_train, y_train)

    print("\nRESULTS ON TRAINING SET")
    print("=" * 60)
    print(f"Alpha — MSE: {results['mse_alpha']:.6f}, R²: {results['r2_alpha']:.4f}")
    print(f"Rho   — MSE: {results['mse_rho']:.6f}, R²: {results['r2_rho']:.4f}")

if __name__ == "__main__":
    main()