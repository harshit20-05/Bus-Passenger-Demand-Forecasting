"""
Historical Route + Weekday + Time-Slot Baseline Model
=====================================================
Implements a rule-based statistical baseline that predicts the historical average
passenger demand for a given (route_id, day_of_week, departure_time_slot).

Fallback Hierarchy:
-------------------
1. Specific: (route_id, day_of_week, time_slot)
2. Route + Weekday: (route_id, day_of_week)
3. Route baseline: (route_id)
4. Global training baseline mean

Time-slot definitions:
- Morning_Peak: 07:00 - 10:59
- Midday_Lull:  11:00 - 15:59
- Evening_Peak: 16:00 - 20:59
- Night:        21:00 - 06:59
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict, Tuple, Any, Optional


def assign_time_slot(hour: int) -> str:
    """Categorizes departure hour into operational time slot."""
    if 7 <= hour <= 10:
        return "Morning_Peak"
    elif 11 <= hour <= 15:
        return "Midday_Lull"
    elif 16 <= hour <= 20:
        return "Evening_Peak"
    else:
        return "Night"


class HistoricalTimeSlotBaseline(BaseEstimator, RegressorMixin):
    """
    Statistical heuristic baseline using historical average passenger count
    conditioned on Route, Day-of-Week, and Time-Slot.
    """
    def __init__(self):
        self.group_means_: Dict[Tuple[str, int, str], float] = {}
        self.route_dow_means_: Dict[Tuple[str, int], float] = {}
        self.route_means_: Dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits historical lookup tables strictly on training data."""
        df = X.copy()
        df["target"] = y.values

        if "time_slot" not in df.columns:
            df["time_slot"] = df["departure_hour"].apply(assign_time_slot)

        # 1. Primary lookup: (route_id, day_of_week, time_slot)
        primary = df.groupby(["route_id", "day_of_week", "time_slot"])["target"].mean()
        self.group_means_ = primary.to_dict()

        # 2. Level 2 fallback: (route_id, day_of_week)
        secondary = df.groupby(["route_id", "day_of_week"])["target"].mean()
        self.route_dow_means_ = secondary.to_dict()

        # 3. Level 3 fallback: route_id
        route_lvl = df.groupby("route_id")["target"].mean()
        self.route_means_ = route_lvl.to_dict()

        # 4. Level 4 fallback: global mean
        self.global_mean_ = float(df["target"].mean())

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts passenger count for input records using hierarchical lookup."""
        df = X.copy()
        if "time_slot" not in df.columns:
            df["time_slot"] = df["departure_hour"].apply(assign_time_slot)

        preds = []
        for _, row in df.iterrows():
            route = row["route_id"]
            dow = row["day_of_week"]
            slot = row["time_slot"]

            # Try primary
            key_primary = (route, dow, slot)
            if key_primary in self.group_means_:
                pred = self.group_means_[key_primary]
            elif (route, dow) in self.route_dow_means_:
                pred = self.route_dow_means_[(route, dow)]
            elif route in self.route_means_:
                pred = self.route_means_[route]
            else:
                pred = self.global_mean_

            preds.append(max(0.0, float(pred)))

        return np.array(preds)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MAE, RMSE, and WAPE.
    Includes zero-denominator safeguard for WAPE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # WAPE: Weighted Absolute Percentage Error = sum(|y - y_hat|) / sum(y) * 100
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual > 0:
        wape = float((np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100.0)
    else:
        wape = 0.0

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE": round(wape, 2),
    }

