
import pathlib

import config
import models
from utils.data_loader import load_ground_truth
import pandas as pd
# train test split import

def create_model_from_config():
    """
    Create and return a model instance based on the configuration.
    """
    if config.TRAINING_METHOD == 'random_forest':
        from models.random_forest import RandomForestModel
        model = RandomForestModel()
    elif config.TRAINING_METHOD == 'neural_net':
        from models.neural_net import NeuralNetworkModel
        model = NeuralNetworkModel()
    else:
        raise ValueError(f"Unknown TRAINING_METHOD: {config.TRAINING_METHOD}")
    
    return model

def load_training_data():
    """
    Load training data for the model.
    
    Returns:
        X_train: Feature matrix
        y_train: Target values
    """
    features_file = (config.FEATURES_DIR / 'features.csv').resolve()
    full_data_df = pd.read_csv(features_file)
    feature_cols = config.FEATURE_COLUMNS
    ground_truth_df = load_ground_truth(config.SIMULATED_DATA_DIR)

    # Merge features with ground truth
    merged_df = pd.merge(
        full_data_df,
        ground_truth_df,
        on=['tree', 'simulation'],
        how='inner'
    )
    
    # Filter for training trees
    train_trees = sorted(merged_df['tree'].unique())[:config.N_TRAIN_TREES]
    train_df = merged_df[merged_df['tree'].isin(train_trees)]

    X_train = train_df[feature_cols].values
    y_train = train_df[['true_alpha', 'true_rho']].values
    train_groups = train_df['tree'].values

    return X_train, y_train, train_groups


def main():
    """Main function to train models."""
    print("Training selected model...")
    print("=" * 50)
    
    # model = create_model_from_config()


    # Load training data
    X_train, y_train, train_groups = load_training_data()
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    # print head of X_train and y_train
    print("First 5 rows of X_train:")
    print(X_train[:5])
    print("First 5 rows of y_train:")
    print(y_train[:5])
    model = create_model_from_config()
    # Train model
    model.train(X_train, y_train, groups=train_groups)  # groups can be None for this split
    # Validate model

    # Save model pickle
    print("Model training completed.")
    import joblib
    model_file = pathlib.Path("models") / f"{config.TRAINING_METHOD}_model.pkl"
    joblib.dump(model, model_file)
    print(f"Trained model saved to: {model_file}")

    # # Simple validation metric: Mean Squared Error
    # from sklearn.metrics import mean_squared_error
    # mse = mean_squared_error(y_val, val_predictions)
    # print(f"Validation Mean Squared Error: {mse}")



if __name__ == "__main__":
    main()


