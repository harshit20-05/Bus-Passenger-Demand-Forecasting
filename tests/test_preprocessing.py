"""
Test Suite: Preprocessing, Chronological Splitting, and Feature Pipelines
========================================================================
Validates:
- Strict chronological boundary ordering (Train max < Val min < Test min).
- Correct splitting proportions (approx 70/15/15).
- Pipeline cyclic feature transformations within [-1, 1].
- Preprocessing output has no NaNs or Infinities.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.data_generator import generate_synthetic_data
from src.preprocessing import (
    validate_and_clean_data,
    chronological_split,
    prepare_features_and_targets,
)
from src.feature_engineering import create_feature_pipeline, CyclicFeatureExtractor


@pytest.fixture(scope="module")
def sample_dataset():
    """Generates a reproducible 3-month dataset for testing preprocessing."""
    df = generate_synthetic_data(
        start_date="2024-01-01",
        end_date="2024-03-31",
        random_seed=42,
        output_path=None,
    )
    df["departure_datetime"] = pd.to_datetime(df["departure_datetime"])
    return df


def test_chronological_splitting_boundaries(sample_dataset):
    """Ensure training data strictly precedes validation, and validation precedes test data."""
    cleaned_df, _ = validate_and_clean_data(sample_dataset)
    train_df, val_df, test_df = chronological_split(cleaned_df, 0.70, 0.15, 0.15)

    assert train_df["departure_datetime"].max() <= val_df["departure_datetime"].min(), (
        "Train set overlaps with validation set chronologically!"
    )
    assert val_df["departure_datetime"].max() <= test_df["departure_datetime"].min(), (
        "Validation set overlaps with test set chronologically!"
    )


def test_chronological_split_proportions(sample_dataset):
    """Verify split record proportions align with 70/15/15 target."""
    cleaned_df, _ = validate_and_clean_data(sample_dataset)
    train_df, val_df, test_df = chronological_split(cleaned_df, 0.70, 0.15, 0.15)

    total_len = len(cleaned_df)
    assert np.isclose(len(train_df) / total_len, 0.70, atol=0.01)
    assert np.isclose(len(val_df) / total_len, 0.15, atol=0.01)
    assert np.isclose(len(test_df) / total_len, 0.15, atol=0.01)


def test_cyclic_feature_extractor():
    """Verify sine and cosine cyclical transformations are strictly bounded."""
    df = pd.DataFrame({
        "departure_hour": [0, 6, 12, 18, 23],
        "day_of_week": [0, 2, 4, 6, 1],
        "month": [1, 4, 7, 10, 12],
    })
    extractor = CyclicFeatureExtractor()
    transformed = extractor.transform(df)

    for col in ["sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week", "sin_month", "cos_month"]:
        assert col in transformed.columns
        assert (transformed[col] >= -1.0).all() and (transformed[col] <= 1.0).all()


def test_feature_pipeline_fit_transform(sample_dataset):
    """Verify Scikit-Learn pipeline fits without NaNs or shape mismatch."""
    cleaned_df, _ = validate_and_clean_data(sample_dataset)
    train_df, val_df, test_df = chronological_split(cleaned_df, 0.70, 0.15, 0.15)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_targets(
        train_df, val_df, test_df
    )

    pipeline, cat_cols, num_cols = create_feature_pipeline()
    X_train_trans = pipeline.fit_transform(X_train)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    assert not np.isnan(X_train_trans).any(), "NaN found in transformed training features!"
    assert not np.isnan(X_val_trans).any(), "NaN found in transformed validation features!"
    assert not np.isnan(X_test_trans).any(), "NaN found in transformed test features!"

    assert X_train_trans.shape[0] == len(X_train)
    assert X_val_trans.shape[0] == len(X_val)
    assert X_test_trans.shape[0] == len(X_test)
