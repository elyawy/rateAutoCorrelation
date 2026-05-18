"""
LightGBM model for predicting alpha and rho parameters.

Trains two separate regressors (one per target) with GroupKFold CV support.
"""

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error


EARLY_STOPPING_ROUNDS = 50


class LightGBMModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_alpha = None
        self.model_rho = None
        self.params = {}
        self.cv_score = None
        self.best_n_estimators = None

    def set_params(self, **params):
        self.params = params

    def train(self, X, y, groups=None):
        """
        Train two LightGBM regressors (alpha and rho).

        If groups are provided, uses GroupKFold CV with early stopping and stores
        the mean normalized MSE in self.cv_score (for Optuna). MSE is normalized
        by each target's variance in the training fold so alpha and rho contribute
        equally regardless of their different value ranges.

        Without groups, trains final models on all data (no early stopping).
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

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(-1),
        ]

        if groups is not None:
            gkf = GroupKFold(n_splits=2)
            cv_scores = []
            best_iterations = []

            for train_idx, val_idx in gkf.split(X, y, groups):
                X_tr, X_val = X[train_idx], X[val_idx]
                a_tr, a_val = y_alpha[train_idx], y_alpha[val_idx]
                r_tr, r_val = y_rho[train_idx], y_rho[val_idx]

                model_a = lgb.LGBMRegressor(**lgb_params)
                model_r = lgb.LGBMRegressor(**lgb_params)

                model_a.fit(X_tr, a_tr, eval_set=[(X_val, a_val)], callbacks=callbacks)
                model_r.fit(X_tr, r_tr, eval_set=[(X_val, r_val)], callbacks=callbacks)

                preds_a = model_a.predict(X_val)
                preds_r = model_r.predict(X_val)

                # Normalize each target's MSE by its training variance so that
                # alpha and rho contribute equally to the Optuna objective.
                var_a = np.var(a_tr) or 1.0
                var_r = np.var(r_tr) or 1.0
                normalized_mse = (
                    mean_squared_error(a_val, preds_a) / var_a +
                    mean_squared_error(r_val, preds_r) / var_r
                ) / 2
                cv_scores.append(normalized_mse)
                best_iterations.append(
                    (model_a.best_iteration_ + model_r.best_iteration_) / 2
                )

            self.cv_score = np.mean(cv_scores)
            self.best_n_estimators = int(np.mean(best_iterations))
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