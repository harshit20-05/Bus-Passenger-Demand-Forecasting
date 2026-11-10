"""
Data Preprocessing and Chronological Splitting Module
=====================================================
Handles data ingestion, validation, cleaning, and strict chronological train/val/test splitting.

Design Principles:
------------------
1. Strict Chronological Ordering:
   - Prevents temporal leakage.
   - 70% Training | 15% Validation | 15% Testing.
   - Enforces: max(train_dt) < min(val_dt) and max(val_dt) < min(test_dt).
2. Data Quality & Validation:
   - Missing-value verification and imputation strategy.
   - Outlier detection (IQR / Z-score) without dropping valid high-demand peak signals.
   - Duplicate record identification.
3. Feature and Target Separation:
   - Target: `passenger_count`.
   - Features: route and departure metadata available 24 hours prior to departure.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


FEATURE_COLUMNS = [
    "route_id",
    "origin",
    "destination",
    "departure_hour",
    "day_of_week",
    "month",
    "season",
    "is_weekend",
    "is_peak_hour",
    "bus_capacity",
    "historical_route_demand",
]

TARGET_COLUMN = "passenger_count"


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads raw trip dataset and parses datetime fields."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)
    df["departure_datetime"] = pd.to_datetime(df["departure_datetime"])
    return df


def validate_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs data quality checks, handles missing values, and identifies duplicates.

    Returns:
    --------
    Tuple of (cleaned_df, quality_report_dict)
    """
    cleaned_df = df.copy()

    # 1. Duplicate check
    initial_len = len(cleaned_df)
    duplicates_count = int(cleaned_df.duplicated(subset=["trip_id"]).sum())
    if duplicates_count > 0:
        cleaned_df = cleaned_df.drop_duplicates(subset=["trip_id"]).reset_index(drop=True)

    # 2. Missing value analysis and safe imputation
    missing_counts = cleaned_df.isnull().sum().to_dict()

    # Impute numerical features if any missing values present
    if cleaned_df["historical_route_demand"].isnull().any():
        # Fallback to route-specific median if available, else overall median
        route_medians = cleaned_df.groupby("route_id")["historical_route_demand"].transform("median")
        cleaned_df["historical_route_demand"] = cleaned_df["historical_route_demand"].fillna(route_medians)
        cleaned_df["historical_route_demand"] = cleaned_df["historical_route_demand"].fillna(
            cleaned_df["historical_route_demand"].median()
        )

    # 3. Sort chronologically
    cleaned_df = cleaned_df.sort_values(by="departure_datetime").reset_index(drop=True)

    # 4. Outlier analysis summary (IQR method on passenger_count)
    q25 = cleaned_df[TARGET_COLUMN].quantile(0.25)
    q75 = cleaned_df[TARGET_COLUMN].quantile(0.75)
    iqr = q75 - q25
    lower_bound = max(0, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr
    outliers_count = int(((cleaned_df[TARGET_COLUMN] < lower_bound) | (cleaned_df[TARGET_COLUMN] > upper_bound)).sum())

    quality_report = {
        "initial_rows": initial_len,
        "retained_rows": len(cleaned_df),
        "duplicate_rows_removed": duplicates_count,
        "missing_values": missing_counts,
        "passenger_count_q25": float(q25),
        "passenger_count_q75": float(q75),
        "passenger_count_iqr": float(iqr),
        "outlier_bounds": (float(lower_bound), float(upper_bound)),
        "outliers_detected": outliers_count,
    }

    return cleaned_df, quality_report


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs strict chronological splitting into Train, Validation, and Test sets.

    Parameters:
    -----------
    df : pd.DataFrame, sorted chronologically
    train_ratio : float, default 0.70
    val_ratio : float, default 0.15
    test_ratio : float, default 0.15

    Returns:
    --------
    (train_df, val_df, test_df)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    # Enforce sorting
    df = df.sort_values(by="departure_datetime").reset_index(drop=True)
    n = len(df)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    # Verification of chronological boundary
    train_max_date = train_df["departure_datetime"].max()
    val_min_date = val_df["departure_datetime"].min()
    val_max_date = val_df["departure_datetime"].max()
    test_min_date = test_df["departure_datetime"].min()

    assert train_max_date <= val_min_date, (
        f"Temporal leakage detected! Train max ({train_max_date}) > Val min ({val_min_date})"
    )
    assert val_max_date <= test_min_date, (
        f"Temporal leakage detected! Val max ({val_max_date}) > Test min ({test_min_date})"
    )

    print(f"Chronological Split Summary:")
    print(f"  Train : {len(train_df):,} trips ({train_df['departure_datetime'].min()} -> {train_max_date})")
    print(f"  Val   : {len(val_df):,} trips ({val_min_date} -> {val_max_date})")
    print(f"  Test  : {len(test_df):,} trips ({test_min_date} -> {test_df['departure_datetime'].max()})")

    return train_df, val_df, test_df


def prepare_features_and_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extracts features X and target y for each partition."""
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_val = val_df[FEATURE_COLUMNS].copy()
    y_val = val_df[TARGET_COLUMN].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test
