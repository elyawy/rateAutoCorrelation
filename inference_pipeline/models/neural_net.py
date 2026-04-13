"""
Neural Network model for predicting alpha and rho parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterPredictionNN(nn.Module):
    """
    Neural network that predicts alpha and rho with enforced bounds.
    
    Output ranges:
    - Alpha: [0.1, 2.0]
    - Rho: [0.01, 0.95]
    """
    def __init__(self, input_size):
        super(ParameterPredictionNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # Output: [alpha, rho]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class NeuralNetworkModel:
    """Wrapper class for training and using the neural network."""
    
    def __init__(self, input_size, learning_rate=0.001, epochs=100, batch_size=32):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = ParameterPredictionNN(input_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, X, y, X_val=None, y_val=None, trial=None):
        """
        Train the neural network.

        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features (required for pruning)
            y_val: Optional validation targets (required for pruning)
            trial: Optional Optuna trial object. If provided alongside validation
                   data, reports val loss each epoch and prunes unpromising trials.
        """
        import optuna

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Prepare validation tensors if provided
        pruning_enabled = trial is not None and X_val is not None and y_val is not None
        if pruning_enabled:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)

        self.model.train()
        for epoch in range(self.epochs):
            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Report validation loss to Optuna and check for pruning
            if pruning_enabled:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val_tensor)
                    val_loss = self.criterion(val_preds, y_val_tensor).item()
                self.model.train()

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            elif (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}]")

    def predict(self, X):
        """Make predictions with the trained model."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        return outputs.numpy()