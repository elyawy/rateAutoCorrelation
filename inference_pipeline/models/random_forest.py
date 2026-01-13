"""
Random Forest model with GroupKFold cross-validation support.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import numpy as np


class RandomForestModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.params = {}
        self.cv_score = None

    def set_params(self, **params):
        """Set hyperparameters for the model."""
        self.params = params

    def train(self, X, y, groups=None):
        """
        Train the Random Forest model.
        
        If groups are provided, uses GroupKFold cross-validation and stores
        the average CV score in self.cv_score (useful for Optuna).
        
        If groups are None, trains a single model on all data.
        """
        if groups is not None:
            # Use GroupKFold for hyperparameter optimization
            gkf = GroupKFold(n_splits=5)
            cv_scores = []
            
            for train_idx, val_idx in gkf.split(X, y, groups):
                model = RandomForestRegressor(
                    random_state=self.random_state, 
                    n_jobs=-1, 
                    **self.params
                )
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[val_idx])
                
                # Calculate MSE for this fold
                mse = mean_squared_error(y[val_idx], preds)
                cv_scores.append(mse)
            
            # Store average CV score
            self.cv_score = np.mean(cv_scores)
            
            # Don't keep any specific fold's model
            # (caller should retrain on full data for final model)
            self.model = None
            
        else:
            # Train single model on all data (for final model or non-CV training)
            self.model = RandomForestRegressor(
                random_state=self.random_state, 
                n_jobs=-1, 
                **self.params
            )
            self.model.fit(X, y)

    def predict(self, X):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet, or was trained with CV only. "
                "Call train() without groups to train a single model."
            )
        return self.model.predict(X)