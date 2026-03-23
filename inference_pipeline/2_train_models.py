"""
Step 2: Train models with hyperparameter optimization.

Loads features, runs Optuna for hyperparameter tuning, trains final model.
"""

import pathlib
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import config

def load_training_data():
    """
    Load training data for the model from the precomputed features file.

    This function reads the feature dataset from ``config.FEATURES_DIR / 'features.csv'``,
    selects the first ``config.N_TRAIN_TREES`` unique trees to define the training
    subset, and constructs:

    - a feature matrix using ``config.FEATURE_COLUMNS``,
    - target values consisting of the columns ``'true_alpha'`` and ``'true_rho'``, and
    - group labels based on the tree identifiers (for use with grouped CV strategies).

    Returns:
        tuple:
            X_train (numpy.ndarray): Feature matrix for training samples.
            y_train (numpy.ndarray): Target values for training samples with
                columns ``true_alpha`` and ``true_rho``.
            train_groups (numpy.ndarray): Group labels (tree IDs) for each
                training sample.
    """
    features_file = (config.FEATURES_DIR / 'features.csv').resolve()
    merged_df = pd.read_csv(features_file)
    print(merged_df[config.FEATURE_COLUMNS].isnull().sum())
    print((merged_df[config.FEATURE_COLUMNS] == np.inf).sum())
    print(merged_df[config.FEATURE_COLUMNS].describe())
    merged_df = merged_df.dropna(subset=config.FEATURE_COLUMNS)
    
    train_trees = sorted(merged_df['tree'].unique())[:config.N_TRAIN_TREES]
    train_df = merged_df[merged_df['tree'].isin(train_trees)]

    X_train = train_df[config.FEATURE_COLUMNS].values
    y_train = train_df[['true_alpha', 'true_rho']].values
    train_groups = train_df['tree'].values

    return X_train, y_train, train_groups

def objective_random_forest(trial, X, y, groups):
    """
    Optuna objective for Random Forest using GroupKFold.
    
    Returns the average cross-validation MSE.
    """
    from models.random_forest import RandomForestModel
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
    }
    
    model = RandomForestModel(random_state=config.MASTER_SEED)
    model.set_params(**params)
    model.train(X, y, groups=groups)
    
    return model.cv_score
    

def objective_neural_net(trial, X, y):
    """
    Optuna objective for Neural Network using simple train/val split.
    
    Returns validation MSE.
    """
    from models.neural_net import NeuralNetworkModel
    
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 50, 200),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }
    
    # Split data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.MASTER_SEED
    )
    
    # Create and train model
    model = NeuralNetworkModel(
        input_size=X_tr.shape[1],
        **params
    )
    model.train(X_tr, y_tr)
    
    # Evaluate on validation set
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    
    return mse


def objective_lightgbm(trial, X, y, groups):
    """
    Optuna objective for LightGBM using GroupKFold.

    Tunes the most impactful hyperparameters first:
    num_leaves, learning_rate, n_estimators, min_child_samples,
    subsample, colsample_bytree.

    Returns the average cross-validation MSE (mean of alpha and rho MSE).
    """
    from models.lightgbm_model import LightGBMModel

    params = {
        'num_leaves':        trial.suggest_int('num_leaves', 20, 300),
        'learning_rate':     trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    model = LightGBMModel(random_state=config.MASTER_SEED)
    model.set_params(**params)
    model.train(X, y, groups=groups)

    return model.cv_score


def train_final_model(best_params, X, y, groups, input_size):
    """
    Train the final model with best hyperparameters on full training data.
    """
    if config.TRAINING_METHOD == 'random_forest':
        from models.random_forest import RandomForestModel
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestModel(random_state=config.MASTER_SEED)
        model.set_params(**best_params)
        model.model = RandomForestRegressor(
            random_state=config.MASTER_SEED, 
            n_jobs=-1, 
            **best_params
        )
        model.model.fit(X, y)
        
    elif config.TRAINING_METHOD == 'neural_net':
        from models.neural_net import NeuralNetworkModel
        
        model = NeuralNetworkModel(
            input_size=input_size,
            **best_params
        )
        model.train(X, y)

    elif config.TRAINING_METHOD == 'lightgbm':
        from models.lightgbm_model import LightGBMModel

        model = LightGBMModel(random_state=config.MASTER_SEED)
        model.set_params(**best_params)
        model.train(X, y)  # No groups = train on full data

    else:
        raise ValueError(f"Unknown TRAINING_METHOD: {config.TRAINING_METHOD}")
    
    return model


def main():
    """Main function to train models with hyperparameter optimization."""
    print("=" * 60)
    print(f"Training {config.TRAINING_METHOD.upper()} model")
    print("=" * 60)
    
    # Load training data ONCE
    print("\nLoading training data...")
    X_train, y_train, train_groups = load_training_data()
    X_train = np.asarray(X_train)  # add this line
    y_train = np.asarray(y_train)
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Number of unique trees: {len(set(train_groups))}")
    
    # Run Optuna optimization
    print(f"\nRunning Optuna optimization ({config.N_OPTUNA_TRIALS} trials)...")
    print("-" * 60)
    
    study = optuna.create_study(direction='minimize')
    
    if config.TRAINING_METHOD == 'random_forest':
        study.optimize(
            lambda trial: objective_random_forest(trial, X_train, y_train, train_groups),
            n_trials=config.N_OPTUNA_TRIALS,
            show_progress_bar=True
        )
    elif config.TRAINING_METHOD == 'neural_net':
        study.optimize(
            lambda trial: objective_neural_net(trial, X_train, y_train),
            n_trials=config.N_OPTUNA_TRIALS,
            show_progress_bar=True
        )
    elif config.TRAINING_METHOD == 'lightgbm':
        study.optimize(
            lambda trial: objective_lightgbm(trial, X_train, y_train, train_groups),
            n_trials=config.N_OPTUNA_TRIALS,
            show_progress_bar=True
        )
    else:
        raise ValueError(f"Unknown TRAINING_METHOD: {config.TRAINING_METHOD}")
    
    print("\n" + "-" * 60)
    print("Optimization complete!")
    print(f"Best validation MSE: {study.best_value:.6f}")
    print(f"Best hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Train final model on full training data
    print("\n" + "-" * 60)
    print("Training final model with best hyperparameters...")
    
    final_model = train_final_model(
        study.best_params, 
        X_train, 
        y_train, 
        train_groups,
        input_size=X_train.shape[1]
    )
    
    # Save model and hyperparameters
    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_file = models_dir / f"{config.TRAINING_METHOD}_model.pkl"
    joblib.dump(final_model, model_file)
    print(f"Model saved to: {model_file}")
    
    # Save best hyperparameters for reference
    params_file = models_dir / f"{config.TRAINING_METHOD}_best_params.pkl"
    joblib.dump(study.best_params, params_file)
    print(f"Best params saved to: {params_file}")
    
    # Save Optuna study for later analysis
    study_file = models_dir / f"{config.TRAINING_METHOD}_study.pkl"
    joblib.dump(study, study_file)
    print(f"Optuna study saved to: {study_file}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()