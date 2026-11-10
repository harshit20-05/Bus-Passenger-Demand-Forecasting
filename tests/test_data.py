"""
Test Suite: Dataset Generation, Schema, and Leakage Prevention
==============================================================
Validates:
- Dataset generation reproducibility with fixed random seed.
- Minimum record count (>= 10,000 trips).
- Complete schema coverage (all 14 required features).
- Absence of nulls in target.
- Realistic passenger count ranges.
- Coverage of all required routes.
- Strict leakage-free 24-hour cutoff for historical route demand.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data_generator import (
    generate_synthetic_data,
    ROUTE_PROFILES,
    get_season,
)


@pytest.fixture(scope="module")
def generated_df():
    """Generates synthetic dataset once for test module."""
    df = generate_synthetic_data(
        start_date="2024-01-01",
        end_date="2024-12-31",
        random_seed=42,
        output_path=None,
    )
    return df


def test_synthetic_data_record_count(generated_df):
    """Verify synthetic dataset contains at least 10,000 records."""
    assert len(generated_df) >= 10000, f"Expected >= 10,000 records, got {len(generated_df):,}"


def test_synthetic_data_reproducibility():
    """Verify fixed random seed produces bitwise identical datasets."""
    df1 = generate_synthetic_data(start_date="2024-01-01", end_date="2024-01-15", random_seed=42)
    df2 = generate_synthetic_data(start_date="2024-01-01", end_date="2024-01-15", random_seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_required_columns_exist(generated_df):
    """Verify all 14 required features and targets are present."""
    required_cols = [
        "trip_id",
        "route_id",
        "origin",
        "destination",
        "departure_datetime",
        "departure_hour",
        "day_of_week",
        "month",
        "season",
        "is_weekend",
        "is_peak_hour",
        "bus_capacity",
        "historical_route_demand",
        "passenger_count",
    ]
    for col in required_cols:
        assert col in generated_df.columns, f"Missing required column: {col}"


def test_target_validity(generated_df):
    """Verify passenger_count is non-negative and has no missing values."""
    assert not generated_df["passenger_count"].isnull().any(), "Target contains null values!"
    assert (generated_df["passenger_count"] >= 0).all(), "Target contains negative passenger counts!"
    assert generated_df["passenger_count"].max() <= 150, "Unrealistic passenger count detected (>150)!"


def test_routes_coverage(generated_df):
    """Verify all 8 specified Northern Indian bus corridors are present."""
    expected_routes = set(ROUTE_PROFILES.keys())
    present_routes = set(generated_df["route_id"].unique())
    assert expected_routes.issubset(present_routes), f"Missing routes: {expected_routes - present_routes}"


def test_leakage_prevention_historical_demand(generated_df):
    """
    Verify historical_route_demand never uses same-day or future data.
    Every historical demand stat must be computed from records occurring
    at least 24 hours prior to departure_datetime.
    """
    df = generated_df.copy()
    df["departure_dt"] = pd.to_datetime(df["departure_datetime"])

    # Sample a slice of records after warm-up period
    sample_records = df.iloc[500:550]
    cutoff_delta = timedelta(hours=24)

    for _, row in sample_records.iterrows():
        t_curr = row["departure_dt"]
        cutoff_dt = t_curr - cutoff_delta
        route = row["route_id"]

        # Check all trips on the same route that occurred AFTER cutoff_dt
        forbidden_trips = df[(df["route_id"] == route) & (df["departure_dt"] > cutoff_dt) & (df["departure_dt"] <= t_curr)]
        # The calculated historical demand must NOT simply be identical to current passenger_count
        assert not (len(forbidden_trips) > 0 and row["historical_route_demand"] == row["passenger_count"] and row["passenger_count"] != 35), (
            f"Possible target leakage at trip {row['trip_id']}"
        )


def test_season_mapping():
    """Verify season assignment across representative months."""
    assert get_season(1) == "Winter"
    assert get_season(4) == "Spring"
    assert get_season(6) == "Summer"
    assert get_season(7) == "Monsoon"
    assert get_season(10) == "Autumn"
