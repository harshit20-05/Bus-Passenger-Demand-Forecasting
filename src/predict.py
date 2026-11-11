"""
Model Inference and Real-Time Prediction Module
===============================================
Provides production inference interface for 24-hour bus passenger demand forecasting.

Key Design Features:
--------------------
1. Real Model Loading: Loads Joblib pipeline artifact. No mock/hardcoded values.
2. Leakage-Safe Feature Derivation: Computes hour, weekday, season, peak status,
   and historical baseline from route metadata at the 24-hour cutoff.
3. Separation of Raw vs Business Adjustment: Retains raw regression output while
   providing nonnegative integer for business operations.
4. Uncapped Demand & Capacity Alerting: Demand prediction is NOT capped by bus capacity
   to accurately reveal spillover demand and overflow risks.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple

from src.data_generator import get_season


def load_model_artifact(model_path: str = "models/best_model.joblib") -> Dict[str, Any]:
    """Loads and verifies the saved model artifact bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_path}'. "
            "Please execute the training pipeline first using: python src/train.py"
        )
    artifact = joblib.load(model_path)
    return artifact


def validate_inference_inputs(
    route_id: str,
    departure_datetime: Union[str, datetime],
    bus_capacity: Optional[int],
    valid_routes: Dict[str, Any],
) -> Tuple[datetime, int]:
    """
    Validates user inference inputs and returns normalized (departure_dt, bus_capacity).
    """
    if route_id not in valid_routes:
        raise ValueError(
            f"Invalid route_id '{route_id}'. Valid route IDs are: {list(valid_routes.keys())}"
        )

    if isinstance(departure_datetime, str):
        try:
            dep_dt = pd.to_datetime(departure_datetime).to_pydatetime()
        except Exception as e:
            raise ValueError(f"Could not parse departure_datetime '{departure_datetime}': {e}")
    elif isinstance(departure_datetime, (datetime, pd.Timestamp)):
        dep_dt = pd.to_datetime(departure_datetime).to_pydatetime()
    else:
        raise ValueError("departure_datetime must be a string or datetime object.")

    if bus_capacity is None:
        bus_capacity = valid_routes[route_id].get("common_capacity", 45)
    else:
        if not isinstance(bus_capacity, (int, np.integer)) or bus_capacity <= 0:
            raise ValueError(f"bus_capacity must be a positive integer, received: {bus_capacity}")

    return dep_dt, int(bus_capacity)


def predict_demand(
    route_id: str,
    departure_datetime: Union[str, datetime],
    bus_capacity: Optional[int] = None,
    model_path: str = "models/best_model.joblib",
) -> Dict[str, Any]:
    """
    Generates real-time passenger demand forecast 24 hours prior to departure.

    Returns:
    --------
    dict containing:
      - raw_prediction: float
      - adjusted_prediction: int (non-negative)
      - bus_capacity: int
      - occupancy_percentage: float
      - demand_category: str
      - capacity_alert: str
      - capacity_action: str
      - explanation_factors: list of str
    """
    artifact = load_model_artifact(model_path)
    pipeline = artifact["pipeline"]
    route_stats = artifact.get("route_stats", {})

    dep_dt, capacity = validate_inference_inputs(
        route_id, departure_datetime, bus_capacity, route_stats
    )

    # Derive calendar and temporal attributes
    hour = dep_dt.hour
    day_of_week = dep_dt.weekday()
    month = dep_dt.month
    season = get_season(month)
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_peak_hour = 1 if hour in [7, 8, 9, 10, 17, 18, 19, 20] else 0

    route_info = route_stats[route_id]
    origin = route_info.get("origin", "Unknown")
    destination = route_info.get("destination", "Unknown")
    historical_demand = route_info.get("historical_mean", 35.0)

    # Construct single-row DataFrame matching the model's expected schema
    input_data = pd.DataFrame([{
        "route_id": route_id,
        "origin": origin,
        "destination": destination,
        "departure_hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "season": season,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "bus_capacity": capacity,
        "historical_route_demand": float(historical_demand),
    }])

    # Raw model inference
    raw_pred = float(pipeline.predict(input_data)[0])

    # Business adjustment: demand cannot be negative
    # Notice we DO NOT artificially cap at bus_capacity, preserving true latent demand!
    adjusted_pred = max(0, int(np.round(raw_pred)))

    # Utilization and capacity alerts
    occupancy_pct = round((adjusted_pred / capacity) * 100.0, 1)

    if occupancy_pct > 100.0:
        demand_category = "Overcapacity (Spillover Risk)"
        alert_level = "CRITICAL_OVERFLOW"
        capacity_alert = (
            f"Overcapacity Warning! Forecasted demand ({adjusted_pred} pax) exceeds bus capacity "
            f"({capacity} seats) by {adjusted_pred - capacity} passengers ({occupancy_pct}% occupancy)."
        )
        capacity_action = (
            "Deploy duplicate relief bus, upgrade to a 55-seater multi-axle coach, "
            "or close booking window to prevent denied boarding."
        )
    elif occupancy_pct >= 85.0:
        demand_category = "High Utilization"
        alert_level = "OPTIMAL_HIGH"
        capacity_alert = (
            f"High Demand! Expected occupancy is {occupancy_pct}% ({adjusted_pred}/{capacity} seats filled)."
        )
        capacity_action = (
            "Standard scheduled operations recommended. Yield management can apply prime-time pricing."
        )
    elif occupancy_pct >= 50.0:
        demand_category = "Moderate / Balanced"
        alert_level = "BALANCED"
        capacity_alert = (
            f"Healthy Load. Expected occupancy is {occupancy_pct}% ({adjusted_pred}/{capacity} seats filled)."
        )
        capacity_action = (
            "Regular dispatch. Profitable baseline with remaining seats available for last-minute walk-ins."
        )
    else:
        demand_category = "Underutilization"
        alert_level = "UNDERUTILIZED"
        capacity_alert = (
            f"Underutilization Warning! Expected occupancy is only {occupancy_pct}% "
            f"({adjusted_pred}/{capacity} seats filled)."
        )
        capacity_action = (
            "Consider offering promotional discounts, consolidating with an adjacent departure, "
            "or assigning a smaller mini-coach (30-35 seats) to reduce fuel overhead."
        )

    # Business explanatory factors
    explanation_factors = []
    if is_peak_hour:
        explanation_factors.append(f"Departure at {hour:02d}:00 falls within prime rush hours (07-10 or 17-20).")
    else:
        explanation_factors.append(f"Departure at {hour:02d}:00 falls in an off-peak / lull travel period.")

    if is_weekend:
        explanation_factors.append("Weekend departure induces higher intercity leisure & return passenger traffic.")
    else:
        explanation_factors.append("Weekday departure subject to standard commercial commuter flow.")

    explanation_factors.append(f"Season is {season} (Month {month}), setting the corridor baseline multiplier.")
    explanation_factors.append(
        f"Corridor '{origin} -> {destination}' carries a historical 14-day baseline of ~{historical_demand:.1f} passengers."
    )

    return {
        "route_id": route_id,
        "origin": origin,
        "destination": destination,
        "departure_datetime": dep_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "bus_capacity": capacity,
        "raw_prediction": round(raw_pred, 2),
        "adjusted_prediction": adjusted_pred,
        "occupancy_percentage": occupancy_pct,
        "demand_category": demand_category,
        "alert_level": alert_level,
        "capacity_alert": capacity_alert,
        "capacity_action": capacity_action,
        "explanation_factors": explanation_factors,
        "champion_model_name": artifact.get("champion_model_name", "Unknown"),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Bus Passenger-Demand Forecasting Inference CLI")
    parser.add_argument("--route", type=str, default="DEL-CHD", help="Route Corridor ID (e.g. DEL-CHD, CHD-MNL)")
    parser.add_argument("--departure", type=str, default="2025-06-20 08:30:00", help="Departure datetime (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--capacity", type=int, default=45, help="Bus vehicle capacity (seats)")
    parser.add_argument("--model", type=str, default="models/best_model.joblib", help="Path to trained joblib artifact")

    args = parser.parse_args()

    print(f"\n--- Running Inference for Route: {args.route} at {args.departure} (Capacity: {args.capacity}) ---")
    prediction = predict_demand(
        route_id=args.route,
        departure_datetime=args.departure,
        bus_capacity=args.capacity,
        model_path=args.model,
    )

    print("\n--- Forecast Results ---")
    for key, val in prediction.items():
        if key == "explanation_factors":
            print("  Explanation Factors:")
            for factor in val:
                print(f"    - {factor}")
        else:
            print(f"  {key}: {val}")
