import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Rho: 0.01 to 0.95
        rho = 0.01 + 0.94 * torch.sigmoid(raw_rho)

        
        return torch.cat((alpha, rho), dim=1)

class NeuralNetworkModel:
    def __init__(self, input_size, learning_rate=0.001, epochs=100, batch_size=32):
        self.model = CorrectedNN(input_size)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def train(self, X, y):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            epoch_loss /= len(dataloader.dataset)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.numpy()