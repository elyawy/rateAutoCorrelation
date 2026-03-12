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
        
        # Fixed architecture (could be made dynamic in the future)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # Output: [alpha, rho]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        raw_output = self.fc4(x)
        
        # Split outputs
        raw_alpha = raw_output[:, 0].unsqueeze(1)
        raw_rho = raw_output[:, 1].unsqueeze(1)
        
        # Apply transformations to enforce bounds        
        return torch.cat((raw_alpha, raw_rho), dim=1)


class NeuralNetworkModel:
    """Wrapper class for training and using the neural network."""
    
    def __init__(self, input_size, learning_rate=0.001, epochs=100, batch_size=32):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training on device:", self.device)
        self.model = ParameterPredictionNN(input_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, X, y):
        """Train the neural network."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
            
            epoch_loss /= len(dataloader.dataset)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
    
    def predict(self, X):
        """Make predictions with the trained model."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        return outputs.cpu().numpy()