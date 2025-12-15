from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

class RandomForestModel():
    def __init__(self, random_state=42):
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [10, 20, 30],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt']
        }
        # Base model
        base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        
        # GroupKFold to keep trees together (5-fold CV)
        gkf = GroupKFold(n_splits=5)

            # Grid search
        self.grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=gkf,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
    
        self.best_model = None


    def train(self, X, y):
        # Implement training logic here
       self.grid_search.fit(X, y)
       self.best_model = self.grid_search.best_estimator_

    def predict(self, X):
        # Implement prediction logic here
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.best_model.predict(X)
