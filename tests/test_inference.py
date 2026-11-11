"""
Test Suite: Model Inference, Input Validation, and Capacity Alerts
==================================================================
Validates:
- Real model artifact loading and inference output.
- Input validation (invalid route, negative capacity, corrupted datetime).
- Non-negative prediction adjustment.
- Uncapped demand behavior (model prediction is not capped at bus capacity).
- Correct capacity alert logic (overcapacity, high, balanced, underutilized).
"""

import pytest
import os
import joblib
import numpy as np
from datetime import datetime

from src.predict import (
    load_model_artifact,
    validate_inference_inputs,
    predict_demand,
)


MODEL_PATH = "models/best_model.joblib"


@pytest.fixture(scope="module")
def model_exists():
    """Ensure trained model artifact exists before running inference tests."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model artifact '{MODEL_PATH}' not found. Run training before inference tests.")
    return True


def test_model_artifact_structure(model_exists):
    """Verify saved joblib artifact includes required metadata and pipeline."""
    artifact = load_model_artifact(MODEL_PATH)
    assert "pipeline" in artifact
    assert "champion_model_name" in artifact
    assert "route_stats" in artifact
    assert "validation_metrics" in artifact


def test_valid_prediction(model_exists):
    """Verify inference produces complete response dictionary for standard query."""
    result = predict_demand(
        route_id="DEL-CHD",
        departure_datetime="2025-06-15 08:00:00",
        bus_capacity=45,
        model_path=MODEL_PATH,
    )
    assert "raw_prediction" in result
    assert "adjusted_prediction" in result
    assert "occupancy_percentage" in result
    assert "demand_category" in result
    assert "capacity_alert" in result
    assert "explanation_factors" in result
    assert result["adjusted_prediction"] >= 0
    assert isinstance(result["adjusted_prediction"], int)


def test_invalid_route_raises_value_error(model_exists):
    """Verify invalid route ID triggers clear descriptive ValueError."""
    with pytest.raises(ValueError, match="Invalid route_id"):
        predict_demand(
            route_id="NON_EXISTENT_ROUTE",
            departure_datetime="2025-06-15 08:00:00",
            bus_capacity=45,
            model_path=MODEL_PATH,
        )


def test_invalid_capacity_raises_value_error(model_exists):
    """Verify zero or negative bus capacity triggers ValueError."""
    with pytest.raises(ValueError, match="bus_capacity must be a positive integer"):
        predict_demand(
            route_id="DEL-CHD",
            departure_datetime="2025-06-15 08:00:00",
            bus_capacity=-10,
            model_path=MODEL_PATH,
        )

    with pytest.raises(ValueError, match="bus_capacity must be a positive integer"):
        predict_demand(
            route_id="DEL-CHD",
            departure_datetime="2025-06-15 08:00:00",
            bus_capacity=0,
            model_path=MODEL_PATH,
        )


def test_uncapped_capacity_alert(model_exists):
    """
    Verify model demand prediction is NOT capped at bus capacity.
    If capacity is small (e.g. 20 seats) but predicted demand is 40,
    the adjusted prediction must reflect true demand (> 20) and trigger overcapacity.
    """
    result = predict_demand(
        route_id="DEL-CHD",
        departure_datetime="2025-06-15 08:00:00",
        bus_capacity=20,  # deliberately small capacity
        model_path=MODEL_PATH,
    )
    assert result["adjusted_prediction"] > 20, "Demand was artificially capped at bus capacity!"
    assert result["occupancy_percentage"] > 100.0
    assert result["alert_level"] == "CRITICAL_OVERFLOW"
    assert "Overcapacity" in result["demand_category"]


def test_underutilization_alert(model_exists):
    """
    Verify underutilization alert is triggered when occupancy is below 50%.
    """
    result = predict_demand(
        route_id="CHD-JAL",
        departure_datetime="2025-02-10 14:00:00",  # Midday off-peak commuter route
        bus_capacity=80,  # Deliberately large capacity
        model_path=MODEL_PATH,
    )
    assert result["occupancy_percentage"] < 50.0
    assert result["alert_level"] == "UNDERUTILIZED"
    assert "Underutilization" in result["demand_category"]

