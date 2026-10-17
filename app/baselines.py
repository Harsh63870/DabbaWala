"""
app/baselines.py
Baseline ML predictors for delivery time ETA:
1. Linear Regression (scikit-learn)
2. XGBoost Regressor
"""

import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import sentry_sdk

CATEGORICAL_FEATURES = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
NUMERIC_FEATURES = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

class BasePredictor:
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.residual_std_ = 4.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_late_probability(self, X: pd.DataFrame, promised_times: np.ndarray) -> np.ndarray:
        """
        Calculates probability P(Delivery_Time > promised_time) using normal distribution over residuals.
        """
        preds = self.predict(X)
        from scipy.stats import norm
        z = (preds - promised_times) / max(self.residual_std_, 1e-3)
        # Normal cdf gives probability that value is <= 0; 1 - norm.cdf(-z) = norm.cdf(z)
        probs = norm.cdf(z)
        return np.clip(probs, 0.01, 0.99)


class LinearRegressionPredictor(BasePredictor):
    def __init__(self):
        super().__init__("LinearRegression")
        self.model = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_clean = X[ALL_FEATURES]
        t0 = time.time()
        self.model.fit(X_clean, y)
        preds = self.model.predict(X_clean)
        residuals = y - preds
        self.residual_std_ = float(np.std(residuals))
        self.is_fitted = True
        return time.time() - t0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LinearRegressionPredictor is not fitted yet.")
        X_clean = X[ALL_FEATURES]
        return np.maximum(5.0, self.model.predict(X_clean))


class XGBoostPredictor(BasePredictor):
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.1):
        super().__init__("XGBoost")
        self.preprocessor = create_preprocessor()
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_clean = X[ALL_FEATURES]
        t0 = time.time()
        X_trans = self.preprocessor.fit_transform(X_clean)
        self.model.fit(X_trans, y)
        preds = self.model.predict(X_trans)
        residuals = y - preds
        self.residual_std_ = float(np.std(residuals))
        self.is_fitted = True
        return time.time() - t0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("XGBoostPredictor is not fitted yet.")
        X_clean = X[ALL_FEATURES]
        X_trans = self.preprocessor.transform(X_clean)
        return np.maximum(5.0, self.model.predict(X_trans))
