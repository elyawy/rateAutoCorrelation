"""
LightGBM model for predicting alpha and rho parameters.

Trains two separate regressors (one per target) with GroupKFold CV support.
"""

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error


class LightGBMModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_alpha = None
        self.model_rho = None
        self.params = {}
        self.cv_score = None

    def set_params(self, **params):
        self.params = params

    def train(self, X, y, groups=None):
        """
        Train two LightGBM regressors (alpha and rho).

        If groups are provided, uses GroupKFold CV and stores average score
        in self.cv_score (for Optuna). Otherwise trains final models on all data.
        """
        X = np.asarray(X)
        y_alpha = y[:, 0]
        y_rho = y[:, 1]

        lgb_params = {
            **self.params,
            'random_state': self.random_state,
            'verbose': -1,
            'n_jobs': -1,
        }

        if groups is not None:
            gkf = GroupKFold(n_splits=2)
            cv_scores = []

            for train_idx, val_idx in gkf.split(X, y, groups):
                model_a = lgb.LGBMRegressor(**lgb_params)
                model_r = lgb.LGBMRegressor(**lgb_params)

                model_a.fit(X[train_idx], y_alpha[train_idx])
                model_r.fit(X[train_idx], y_rho[train_idx])

                preds_a = model_a.predict(X[val_idx])
                preds_r = model_r.predict(X[val_idx])

                mse = (
                    mean_squared_error(y_alpha[val_idx], preds_a) +
                    mean_squared_error(y_rho[val_idx], preds_r)
                ) / 2
                cv_scores.append(mse)

            self.cv_score = np.mean(cv_scores)
            self.model_alpha = None
            self.model_rho = None

        else:
            self.model_alpha = lgb.LGBMRegressor(**lgb_params)
            self.model_rho = lgb.LGBMRegressor(**lgb_params)
            self.model_alpha.fit(X, y_alpha)
            self.model_rho.fit(X, y_rho)

    def predict(self, X):
        """Returns predictions as (n_samples, 2) array: [alpha, rho]."""
        if self.model_alpha is None or self.model_rho is None:
            raise ValueError(
                "Model has not been trained yet, or was trained with CV only. "
                "Call train() without groups to train final models."
            )
        preds_alpha = self.model_alpha.predict(X)
        preds_rho = self.model_rho.predict(X)
        return np.column_stack([preds_alpha, preds_rho])
