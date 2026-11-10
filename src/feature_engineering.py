"""
Feature Engineering Pipeline Module
====================================
Transforms raw trip attributes into predictive representations without temporal or target leakage.

Features & Business Rationale:
--------------------------------
1. `route_id`, `origin`, `destination`:
   - Captures corridor-specific demand baselines, trip length, and regional attractiveness.
2. `departure_hour`:
   - Diurnal commute cycles: Morning peak (07:00-10:00), Evening peak (17:00-20:00), Night express (21:00-23:00).
3. `day_of_week`:
   - Intra-week dynamics: Monday commercial surge, Friday getaway, Sunday return rush.
4. `month` & `season`:
   - Annual macro patterns: Hill station peaks in Summer (May-June) & Snow season (Dec-Jan); Monsoon drops.
5. `is_weekend` & `is_peak_hour`:
   - Discontinuous operational regimes between commuter flow and leisure travel.
6. `historical_route_demand`:
   - Rolling 14-day demand average evaluated strictly prior to the 24-hour departure cutoff.
7. Cyclical representations:
   - `sin_hour`, `cos_hour`, `sin_day_of_week`, `cos_day_of_week`, `sin_month`, `cos_month`
   - Preserves circular continuity (e.g. 23:00 is temporally adjacent to 00:00, December is adjacent to January).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Tuple


class CyclicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Encodes cyclical time features using sine and cosine transformations.
    hour -> 24-hour cycle
    day_of_week -> 7-day cycle
    month -> 12-month cycle
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # Hour cycle (0 to 23)
        if "departure_hour" in X_out.columns:
            X_out["sin_hour"] = np.sin(2 * np.pi * X_out["departure_hour"] / 24.0)
            X_out["cos_hour"] = np.cos(2 * np.pi * X_out["departure_hour"] / 24.0)

        # Day of week cycle (0 to 6)
        if "day_of_week" in X_out.columns:
            X_out["sin_day_of_week"] = np.sin(2 * np.pi * X_out["day_of_week"] / 7.0)
            X_out["cos_day_of_week"] = np.cos(2 * np.pi * X_out["day_of_week"] / 7.0)

        # Month cycle (1 to 12)
        if "month" in X_out.columns:
            X_out["sin_month"] = np.sin(2 * np.pi * (X_out["month"] - 1) / 12.0)
            X_out["cos_month"] = np.cos(2 * np.pi * (X_out["month"] - 1) / 12.0)

        return X_out


def create_feature_pipeline() -> Tuple[Pipeline, List[str], List[str]]:
    """
    Builds a Scikit-Learn ColumnTransformer pipeline for features.

    Categorical:
      - route_id, origin, destination, season -> OneHotEncoder
    Numerical:
      - departure_hour, day_of_week, month, is_weekend, is_peak_hour,
        bus_capacity, historical_route_demand, sin/cos cyclical features
    """
    categorical_cols = ["route_id", "origin", "destination", "season"]
    numeric_cols = [
        "departure_hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_peak_hour",
        "bus_capacity",
        "historical_route_demand",
        "sin_hour",
        "cos_hour",
        "sin_day_of_week",
        "cos_day_of_week",
        "sin_month",
        "cos_month",
    ]

    col_transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline(
        steps=[
            ("cyclic_encoder", CyclicFeatureExtractor()),
            ("column_preprocessor", col_transformer),
        ]
    )

    return full_pipeline, categorical_cols, numeric_cols
