"""
Step 2: Train models with hyperparameter optimization.

Loads features, runs Optuna for hyperparameter tuning, trains final model.

For neural_net, parallel workers can be used:
    python 2_train_models.py --workers 4
"""

import multiprocessing
import pathlib
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def objective_random_forest(trial, X, y, groups):
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
    from models.neural_net import NeuralNetworkModel

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 30, 100),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.MASTER_SEED
    )

    model = NeuralNetworkModel(input_size=X_tr.shape[1], **params)
    model.train(X_tr, y_tr, X_val=X_val, y_val=y_val, trial=trial)

    predictions = model.predict(X_val)
    return mean_squared_error(y_val, predictions)


def objective_lightgbm(trial, X, y, groups):
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

        model = NeuralNetworkModel(input_size=input_size, **best_params)
        model.train(X, y)  # No pruning for final model — train on all data

    elif config.TRAINING_METHOD == 'lightgbm':
        from models.lightgbm_model import LightGBMModel

        model = LightGBMModel(random_state=config.MASTER_SEED)
        model.set_params(**best_params)
        model.train(X, y)

    else:
        raise ValueError(f"Unknown TRAINING_METHOD: {config.TRAINING_METHOD}")

    return model


def run_nn_worker(study_name, storage_url, X, y, n_trials):
    """
    Worker function for parallel NN Optuna optimization.
    Each worker loads the shared study and contributes trials independently.
    """
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
    )
    study.optimize(
        lambda trial: objective_neural_net(trial, X, y),
        n_trials=n_trials,
        callbacks=[lambda study, trial: print(
            f"  Trial {trial.number} finished — val MSE: {trial.value:.6f}"
            if trial.value is not None else
            f"  Trial {trial.number} pruned"
        )],
    )


def main():
    parser = argparse.ArgumentParser(description='Train models with hyperparameter optimization')
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of parallel Optuna workers (neural_net only, default: 1)'
    )
    args = parser.parse_args()

    if args.workers > 1 and config.TRAINING_METHOD != 'neural_net':
        print(f"WARNING: --workers > 1 is only supported for neural_net. "
              f"Ignoring and using 1 worker for {config.TRAINING_METHOD}.")
        args.workers = 1

    print("=" * 60)
    print(f"Training {config.TRAINING_METHOD.upper()} model")
    if config.TRAINING_METHOD == 'neural_net' and args.workers > 1:
        print(f"Parallel Optuna workers: {args.workers}")
    print("=" * 60)

    print("\nLoading training data...")
    X_train, y_train, train_groups = load_training_data()
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Number of unique trees: {len(set(train_groups))}")

    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------
    # Create study
    # -------------------------------------------------------
    if config.TRAINING_METHOD == 'neural_net':
        storage_url = f"sqlite:///{models_dir}/neural_net_optuna.db"
        study_name = "neural_net_study"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
    else:
        study = optuna.create_study(direction='minimize')

    # -------------------------------------------------------
    # Run optimization
    # -------------------------------------------------------
    print(f"\nRunning Optuna optimization ({config.N_OPTUNA_TRIALS} trials)...")
    print("-" * 60)

    if config.TRAINING_METHOD == 'neural_net' and args.workers > 1:
        # Distribute trials evenly across workers
        trials_per_worker = config.N_OPTUNA_TRIALS // args.workers
        # Any remainder goes to the first worker
        trial_counts = [trials_per_worker + (config.N_OPTUNA_TRIALS % args.workers)] + \
                       [trials_per_worker] * (args.workers - 1)

        processes = []
        for n_trials in trial_counts:
            p = multiprocessing.Process(
                target=run_nn_worker,
                args=(study_name, storage_url, X_train, y_train, n_trials)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Reload study to get final results
        study = optuna.load_study(study_name=study_name, storage=storage_url)

    elif config.TRAINING_METHOD == 'random_forest':
        study.optimize(
            lambda trial: objective_random_forest(trial, X_train, y_train, train_groups),
            n_trials=config.N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )
    elif config.TRAINING_METHOD == 'neural_net':
        study.optimize(
            lambda trial: objective_neural_net(trial, X_train, y_train),
            n_trials=config.N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )
    elif config.TRAINING_METHOD == 'lightgbm':
        study.optimize(
            lambda trial: objective_lightgbm(trial, X_train, y_train, train_groups),
            n_trials=config.N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )
    else:
        raise ValueError(f"Unknown TRAINING_METHOD: {config.TRAINING_METHOD}")

    print("\n" + "-" * 60)
    print("Optimization complete!")
    print(f"Best validation MSE: {study.best_value:.6f}")
    print("Best hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    # -------------------------------------------------------
    # Train final model on full training data
    # -------------------------------------------------------
    print("\n" + "-" * 60)
    print("Training final model with best hyperparameters...")

    final_model = train_final_model(
        study.best_params,
        X_train,
        y_train,
        train_groups,
        input_size=X_train.shape[1]
    )

    model_file = models_dir / f"{config.TRAINING_METHOD}_model.pkl"
    joblib.dump(final_model, model_file)
    print(f"Model saved to: {model_file}")

    params_file = models_dir / f"{config.TRAINING_METHOD}_best_params.pkl"
    joblib.dump(study.best_params, params_file)
    print(f"Best params saved to: {params_file}")

    study_file = models_dir / f"{config.TRAINING_METHOD}_study.pkl"
    joblib.dump(study, study_file)
    print(f"Optuna study saved to: {study_file}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()