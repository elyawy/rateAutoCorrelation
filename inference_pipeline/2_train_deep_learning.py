"""
Step 2: Train Deep Learning models to predict alpha and rho from features.

Trains two separate PyTorch neural networks:
  - One for predicting alpha
  - One for predicting rho

Uses early stopping and validation monitoring.

Input:
  - results/features.csv (statistics)
  - ../simulated_data/{tree}/ground_truth.csv (true alpha/rho values)

Output:
  - results/alpha_model.pt (trained model weights)
  - results/rho_model.pt (trained model weights)
  - results/dl_evaluation.csv (performance metrics)
  - results/training_history.png (loss curves)
"""

import pathlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

# Import local config (inference_pipeline/config.py)
import config as local_config

class CorrectedNN(nn.Module):
    def __init__(self, input_size):
        super(CorrectedNN, self).__init__()
        # Removed Dropout layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        
        # CHANGED: Output 2 values, not 1
        self.fc4 = nn.Linear(32, 2) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) # Removed Dropout
        x = F.relu(self.fc3(x)) # Removed Dropout
        
        raw_output = self.fc4(x)
        
        # Split and Scale (The "Bound Enforcement" Logic)
        raw_alpha = raw_output[:, 0].unsqueeze(1)
        raw_rho   = raw_output[:, 1].unsqueeze(1)
        
        # Alpha: 0.1 to 2.0
        alpha = 0.1 + 1.9 * torch.sigmoid(raw_alpha)
        
        # Rho: 0.05 to 0.95
        rho = 0.05 + 0.9 * torch.sigmoid(raw_rho)
        
        return torch.cat((alpha, rho), dim=1)


class SimpleNN(nn.Module):
    """
    Simple fully connected neural network for regression.
    
    Architecture: Input → 64 → 32 → 1
    """
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Back to 128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # No activation
        return x


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
            
    def load_best_model(self, model):
        model.load_state_dict(self.best_model)


def load_ground_truth(simulated_data_dir):
    """
    Load all ground truth files and combine them.
    
    Returns:
        DataFrame with columns: tree, simulation, true_alpha, true_rho
    """
    all_truth = []
    
    simulated_data_path = pathlib.Path(simulated_data_dir)
    
    for tree_dir in sorted(simulated_data_path.iterdir()):
        if not tree_dir.is_dir():
            continue
        
        tree_name = tree_dir.name
        truth_file = tree_dir / 'ground_truth.csv'
        
        if not truth_file.exists():
            continue
        
        df = pd.read_csv(truth_file)
        
        # Extract simulation name from filename (remove .phy extension)
        df['simulation'] = df['filename'].str.replace('.phy', '', regex=False)
        df['tree'] = tree_name
        
        # Select relevant columns
        df = df[['tree', 'simulation', 'true_alpha', 'true_rho']]
        
        all_truth.append(df)
    
    return pd.concat(all_truth, ignore_index=True)


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=200, patience=20, param_name='parameter'):
    """
    Train a PyTorch model with early stopping.
    
    Returns:
        - Trained model
        - Training history (train/val losses)
    """
    early_stopping = EarlyStopping(patience=patience)
    train_losses = []
    val_losses = []
    
    print(f"  Training {param_name} model...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"    Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    early_stopping.load_best_model(model)
    print(f"    Best validation loss: {early_stopping.best_loss:.6f}")
    
    return model, {'train': train_losses, 'val': val_losses}


def evaluate_model(model, X, y):
    """
    Evaluate model and return predictions.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy().flatten()
    
    return predictions


def calculate_metrics(y_true, y_pred, split_name, param_name):
    """Calculate evaluation metrics."""
    return {
        'split': split_name,
        'parameter': param_name,
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'n_samples': len(y_true)
    }


def plot_training_history(alpha_history, rho_history, output_file):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Alpha plot
    ax1.plot(alpha_history['train'], label='Train Loss', alpha=0.8)
    ax1.plot(alpha_history['val'], label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Alpha Model Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rho plot
    ax2.plot(rho_history['train'], label='Train Loss', alpha=0.8)
    ax2.plot(rho_history['val'], label='Val Loss', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Rho Model Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Training history plot saved to: {output_file}")
    plt.close()


def main():
    """Main function to train deep learning models."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(local_config.MASTER_SEED)
    np.random.seed(local_config.MASTER_SEED)
    
    # Paths
    results_dir = pathlib.Path('results')
    features_file = results_dir / 'features_test.csv'
    simulated_data_dir = local_config.SIMULATED_DATA_DIR
    
    # Check if features file exists
    if not features_file.exists():
        print(f"Error: {features_file} does not exist. Run step 1 first.")
        return
    
    print("Loading data...")
    print("=" * 50)
    
    # Load features
    features_df = pd.read_csv(features_file)
    print(f"Loaded {len(features_df)} feature rows")
    
    # Load ground truth
    ground_truth = load_ground_truth(simulated_data_dir)
    print(f"Loaded {len(ground_truth)} ground truth rows")
    
    # Merge on tree and simulation
    merged = pd.merge(
        features_df,
        ground_truth,
        on=['tree', 'simulation'],
        how='inner'
    )
    
    print(f"Merged dataset: {len(merged)} rows")
    
    if len(merged) == 0:
        print("Error: No matching data found after merge.")
        return
    
    samples_per_tree = local_config.N_SIMS_PER_TREE
    print(f"\nSampling {samples_per_tree} samples per tree...")
    
    sampled_rows = []
    for tree_name in merged['tree'].unique():
        tree_data = merged[merged['tree'] == tree_name]
        if len(tree_data) >= samples_per_tree:
            sampled = tree_data.sample(n=samples_per_tree, random_state=local_config.MASTER_SEED)
        else:
            sampled = tree_data  # Use all if fewer than 100
        sampled_rows.append(sampled)
    # Split trees into train/test sets

    merged = pd.concat(sampled_rows, ignore_index=True)
    print(f"After sampling: {len(merged)} rows")

    all_trees = sorted(merged['tree'].unique())
    n_train_trees = local_config.N_TRAIN_TREES
    
    if n_train_trees >= len(all_trees):
        print(f"\nWarning: N_TRAIN_TREES ({n_train_trees}) >= total trees ({len(all_trees)})")
        print("Using all trees for training. No held-out test set.")
        train_trees = all_trees
        test_trees = []
    else:
        train_trees = all_trees[:n_train_trees]
        test_trees = all_trees[n_train_trees:]
    
    print(f"\nTree split:")
    print(f"  Total trees: {len(all_trees)}")
    print(f"  Training trees: {len(train_trees)} (first {n_train_trees})")
    print(f"  Test trees: {len(test_trees)} (held-out)")
    print(f"\nTraining trees: {train_trees[:5]}{'...' if len(train_trees) > 5 else ''}")
    if test_trees:
        print(f"Test trees: {test_trees[:5]}{'...' if len(test_trees) > 5 else ''}")
    
    # Filter data by tree membership
    train_data = merged[merged['tree'].isin(train_trees)].copy()
    test_data = merged[merged['tree'].isin(test_trees)].copy() if test_trees else None
    
    print(f"\nData split:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data) if test_data is not None else 0}")
    
    # Prepare features and targets
    feature_columns = local_config.FEATURE_COLUMNS
    
    X_train = train_data[feature_columns].values.astype(np.float32)
    y_alpha_train = train_data['true_alpha'].values.astype(np.float32).reshape(-1, 1)
    y_rho_train = train_data['true_rho'].values.astype(np.float32).reshape(-1, 1)
    
    if test_data is not None and len(test_data) > 0:
        X_test = test_data[feature_columns].values.astype(np.float32)
        y_alpha_test = test_data['true_alpha'].values.astype(np.float32).reshape(-1, 1)
        y_rho_test = test_data['true_rho'].values.astype(np.float32).reshape(-1, 1)
    else:
        # No test set - use dummy values
        X_test = X_train[:1]
        y_alpha_test = y_alpha_train[:1]
        y_rho_test = y_rho_train[:1]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nFeatures shape: Train {X_train.shape}, Test {X_test.shape if test_data is not None else 'N/A'}")
    print(f"Feature columns ({len(feature_columns)}):")
    for fc in feature_columns:
        print(f"  - {fc}")
    
    # Hyperparameters
    batch_size = 64
    num_epochs = 200
    patience = 20
    learning_rate = 0.001
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Architecture: {len(feature_columns)} → 64 → 32 → 1")
    
    # Create data loaders
    train_alpha_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_alpha_train)
    )
    train_alpha_loader = DataLoader(
        train_alpha_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # For validation, we'll use a portion of training data (20%)
    val_size = int(0.2 * len(train_data))
    train_size = len(train_data) - val_size
    
    # Split training data into train/val
    indices = np.random.permutation(len(train_data))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Alpha loaders
    train_alpha_dataset = TensorDataset(
        torch.FloatTensor(X_train[train_indices]), 
        torch.FloatTensor(y_alpha_train[train_indices])
    )
    val_alpha_dataset = TensorDataset(
        torch.FloatTensor(X_train[val_indices]), 
        torch.FloatTensor(y_alpha_train[val_indices])
    )
    train_alpha_loader = DataLoader(train_alpha_dataset, batch_size=batch_size, shuffle=True)
    val_alpha_loader = DataLoader(val_alpha_dataset, batch_size=batch_size, shuffle=False)
    
    # Rho loaders
    train_rho_dataset = TensorDataset(
        torch.FloatTensor(X_train[train_indices]), 
        torch.FloatTensor(y_rho_train[train_indices])
    )
    val_rho_dataset = TensorDataset(
        torch.FloatTensor(X_train[val_indices]), 
        torch.FloatTensor(y_rho_train[val_indices])
    )
    train_rho_loader = DataLoader(train_rho_dataset, batch_size=batch_size, shuffle=True)
    val_rho_loader = DataLoader(val_rho_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n" + "=" * 50)
    print("Training Alpha Model")
    print("=" * 50)
    
    # Initialize alpha model
    alpha_model = SimpleNN(input_size=len(feature_columns))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(alpha_model.parameters(), lr=learning_rate)
    
    # Train alpha model
    alpha_model, alpha_history = train_model(
        alpha_model, train_alpha_loader, val_alpha_loader,
        criterion, optimizer, num_epochs, patience, 'Alpha'
    )
    
    # Evaluate alpha model
    y_alpha_train_pred = evaluate_model(alpha_model, X_train, y_alpha_train)
    alpha_train_metrics = calculate_metrics(
        y_alpha_train.flatten(), y_alpha_train_pred, 'train', 'alpha'
    )
    
    print(f"\nAlpha Model Performance:")
    print(f"  Train RMSE: {alpha_train_metrics['rmse']:.4f}")
    print(f"  Train R²: {alpha_train_metrics['r2']:.4f}")
    
    if test_data is not None and len(test_data) > 0:
        y_alpha_test_pred = evaluate_model(alpha_model, X_test, y_alpha_test)
        alpha_test_metrics = calculate_metrics(
            y_alpha_test.flatten(), y_alpha_test_pred, 'test', 'alpha'
        )
        print(f"  Test RMSE (held-out trees): {alpha_test_metrics['rmse']:.4f}")
        print(f"  Test R²: {alpha_test_metrics['r2']:.4f}")
        print(f"  Test MAE: {alpha_test_metrics['mae']:.4f}")
    
    print("\n" + "=" * 50)
    print("Training Rho Model")
    print("=" * 50)
    
    # Initialize rho model
    rho_model = SimpleNN(input_size=len(feature_columns))
    optimizer = optim.Adam(rho_model.parameters(), lr=learning_rate)
    
    # Train rho model
    rho_model, rho_history = train_model(
        rho_model, train_rho_loader, val_rho_loader,
        criterion, optimizer, num_epochs, patience, 'Rho'
    )
    
    # Evaluate rho model
    y_rho_train_pred = evaluate_model(rho_model, X_train, y_rho_train)
    rho_train_metrics = calculate_metrics(
        y_rho_train.flatten(), y_rho_train_pred, 'train', 'rho'
    )
    
    print(f"\nRho Model Performance:")
    print(f"  Train RMSE: {rho_train_metrics['rmse']:.4f}")
    print(f"  Train R²: {rho_train_metrics['r2']:.4f}")
    
    if test_data is not None and len(test_data) > 0:
        y_rho_test_pred = evaluate_model(rho_model, X_test, y_rho_test)
        rho_test_metrics = calculate_metrics(
            y_rho_test.flatten(), y_rho_test_pred, 'test', 'rho'
        )
        print(f"  Test RMSE (held-out trees): {rho_test_metrics['rmse']:.4f}")
        print(f"  Test R²: {rho_test_metrics['r2']:.4f}")
        print(f"  Test MAE: {rho_test_metrics['mae']:.4f}")
    
    # Save models
    print("\n" + "=" * 50)
    print("Saving models...")
    
    alpha_model_path = results_dir / 'alpha_model.pt'
    rho_model_path = results_dir / 'rho_model.pt'
    
    torch.save(alpha_model.state_dict(), alpha_model_path)
    print(f"  Alpha model saved to: {alpha_model_path}")
    
    torch.save(rho_model.state_dict(), rho_model_path)
    print(f"  Rho model saved to: {rho_model_path}")
    
    # Save evaluation metrics
    metrics_list = [alpha_train_metrics, rho_train_metrics]
    if test_data is not None and len(test_data) > 0:
        metrics_list.extend([alpha_test_metrics, rho_test_metrics])
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = results_dir / 'dl_evaluation.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  Evaluation metrics saved to: {metrics_file}")
    
    # Plot training history
    plot_file = results_dir / 'training_history.png'
    plot_training_history(alpha_history, rho_history, plot_file)
    
    # Save tree split information
    tree_split_file = results_dir / 'tree_split.csv'
    tree_split_df = pd.DataFrame({
        'tree': all_trees,
        'split': ['train' if t in train_trees else 'test' for t in all_trees]
    })
    tree_split_df.to_csv(tree_split_file, index=False)
    print(f"  Tree split info saved to: {tree_split_file}")
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
