"""
Synthetic Bus Trip Data Generator
=================================
Generates reproducible, realistic bus trip records for passenger demand forecasting.

Assumptions & Design:
---------------------
1. Synthetic Labeling: Clearly marked as synthetic data for research & prototyping.
2. Route Network: 8 realistic northern Indian intercity bus corridors:
   - Delhi - Chandigarh (Major commercial/institutional corridor)
   - Chandigarh - Manali (High-altitude tourist corridor, strong seasonal surges)
   - Ludhiana - Delhi (Industrial/commercial business corridor)
   - Delhi - Jaipur (Golden Triangle tourist & commercial route)
   - Chandigarh - Shimla (Hill station leisure corridor, weekend surges)
   - Ludhiana - Amritsar (High-frequency commuter & religious corridor)
   - Delhi - Dehradun (Educational hub & tourist gateway)
   - Chandigarh - Jalandhar (Regional commuter route)
3. Demand Mechanics:
   - Base route volume
   - Day of week variations (Friday evening / Sunday evening surges, weekend leisure)
   - Departure hour variations (Morning rush 07:00-10:00, Evening rush 17:00-20:00, Night 21:00-23:00)
   - Seasonal variations (Summer & Winter peaks for hill routes, Monsoon dips)
   - Random noise and occasional outlier events (festivals, localized weather disruptions)
4. Leakage Prevention:
   - Target variable: `passenger_count`
   - `historical_route_demand` is computed using strictly past trips completed AT LEAST
     24 hours before the scheduled departure date/time (the 24-hour prediction cutoff).
   - No post-departure or future features are included.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple


# Route profiles: (origin, destination, base_demand, seasonal_profile, route_type)
ROUTE_PROFILES = {
    "DEL-CHD": {
        "origin": "Delhi",
        "destination": "Chandigarh",
        "base_demand": 42.0,
        "demand_std": 5.0,
        "profile": "commercial_express",
        "distance_km": 250,
        "common_capacities": [45, 50, 55],
    },
    "CHD-MNL": {
        "origin": "Chandigarh",
        "destination": "Manali",
        "base_demand": 38.0,
        "demand_std": 6.5,
        "profile": "tourist_hill",
        "distance_km": 300,
        "common_capacities": [40, 45, 50],
    },
    "LDH-DEL": {
        "origin": "Ludhiana",
        "destination": "Delhi",
        "base_demand": 37.0,
        "demand_std": 5.5,
        "profile": "commercial",
        "distance_km": 315,
        "common_capacities": [45, 50, 55],
    },
    "DEL-JAI": {
        "origin": "Delhi",
        "destination": "Jaipur",
        "base_demand": 41.0,
        "demand_std": 5.0,
        "profile": "tourist_commercial",
        "distance_km": 280,
        "common_capacities": [45, 50, 55],
    },
    "CHD-SML": {
        "origin": "Chandigarh",
        "destination": "Shimla",
        "base_demand": 35.0,
        "demand_std": 6.0,
        "profile": "tourist_hill",
        "distance_km": 115,
        "common_capacities": [40, 45],
    },
    "LDH-ASR": {
        "origin": "Ludhiana",
        "destination": "Amritsar",
        "base_demand": 32.0,
        "demand_std": 4.5,
        "profile": "commuter_cultural",
        "distance_km": 140,
        "common_capacities": [45, 50],
    },
    "DEL-DDN": {
        "origin": "Delhi",
        "destination": "Dehradun",
        "base_demand": 34.0,
        "demand_std": 5.0,
        "profile": "education_tourist",
        "distance_km": 255,
        "common_capacities": [45, 50],
    },
    "CHD-JAL": {
        "origin": "Chandigarh",
        "destination": "Jalandhar",
        "base_demand": 30.0,
        "demand_std": 4.0,
        "profile": "commuter",
        "distance_km": 150,
        "common_capacities": [40, 45, 50],
    },
}

# Standard scheduled departure slots throughout the day
SCHEDULED_HOURS = [6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19, 21, 22]


def get_season(month: int) -> str:
    """Map calendar month to northern Indian climatological/travel season."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4]:
        return "Spring"
    elif month in [5, 6]:
        return "Summer"
    elif month in [7, 8]:
        return "Monsoon"
    else:
        return "Autumn"


def calculate_hourly_multiplier(hour: int, profile: str) -> float:
    """Calculates departure hour demand factor."""
    # Morning rush
    if hour in [7, 8, 9]:
        return 1.25 if "commuter" in profile or "commercial" in profile else 1.15
    elif hour in [10]:
        return 1.10
    # Afternoon lull
    elif hour in [12, 14]:
        return 0.85
    # Late afternoon / Evening rush
    elif hour in [16]:
        return 1.05
    elif hour in [17, 18, 19]:
        return 1.30 if "commercial" in profile or "commuter" in profile else 1.20
    elif hour in [20, 21]:
        return 1.10
    # Night departures (popular for long intercity like Delhi-Manali, Delhi-Ludhiana)
    elif hour in [22, 23]:
        return 1.15 if profile in ["tourist_hill", "commercial_express"] else 0.80
    else:
        return 0.75


def calculate_weekday_multiplier(day_of_week: int, profile: str) -> float:
    """
    Day of week multiplier.
    0 = Monday, ..., 4 = Friday, 5 = Saturday, 6 = Sunday.
    """
    if profile == "tourist_hill":
        # Hill tourists travel Friday evening / Saturday morning, return Sunday
        multipliers = {0: 0.85, 1: 0.80, 2: 0.80, 3: 0.90, 4: 1.30, 5: 1.35, 6: 1.25}
    elif "commercial" in profile:
        # Commercial travel surges Monday morning and Friday evening
        multipliers = {0: 1.25, 1: 1.00, 2: 0.95, 3: 1.00, 4: 1.25, 5: 0.90, 6: 1.10}
    elif "commuter" in profile:
        # Commuter routes heavy on weekdays
        multipliers = {0: 1.15, 1: 1.05, 2: 1.00, 3: 1.05, 4: 1.20, 5: 0.85, 6: 0.90}
    else:
        multipliers = {0: 1.05, 1: 0.95, 2: 0.95, 3: 1.00, 4: 1.15, 5: 1.10, 6: 1.15}
    return multipliers.get(day_of_week, 1.0)


def calculate_seasonal_multiplier(month: int, season: str, profile: str) -> float:
    """Seasonal demand adjustment based on destination characteristics."""
    if profile == "tourist_hill":
        # Manali / Shimla: Extreme peak in Summer (May-June) & Winter snowfall (Dec-Jan)
        # Heavy dip in Monsoon (July-August landslides/rain risk)
        if month in [5, 6]:
            return 1.40  # Summer vacation rush
        elif month in [12, 1]:
            return 1.30  # Snow / New Year rush
        elif month in [7, 8]:
            return 0.65  # Monsoon landslides caution
        elif month in [9, 10]:
            return 1.10  # Post-monsoon pleasant weather
        else:
            return 0.95
    elif profile == "tourist_commercial":
        # Delhi - Jaipur: Peak tourism in winter (Nov-Feb), extreme hot in May-June
        if month in [11, 12, 1, 2]:
            return 1.25
        elif month in [5, 6]:
            return 0.80
        else:
            return 1.00
    elif profile == "education_tourist":
        # Dehradun: Academic sessions start July/August, exam periods, pleasant autumn
        if month in [7, 8]:
            return 1.20
        elif month in [4, 5, 10]:
            return 1.10
        else:
            return 0.95
    else:
        # Standard commercial / commuter routes
        if month in [10, 11]:  # Festival season (Diwali, Dussehra)
            return 1.20
        elif month in [6]:  # Summer break
            return 1.05
        else:
            return 1.00


def generate_synthetic_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    random_seed: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates a reproducible synthetic dataset of at least 10,000 bus trip records.

    Parameters:
    -----------
    start_date : str, ISO format 'YYYY-MM-DD'
    end_date : str, ISO format 'YYYY-MM-DD'
    random_seed : int, fixed seed for full reproducibility
    output_path : Optional[str], path to save the generated CSV

    Returns:
    --------
    pd.DataFrame with trip records and leakage-safe historical demand.
    """
    np.random.seed(random_seed)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    records = []
    trip_counter = 1

    # Generate trips day by day across all routes
    for day_offset in range(total_days):
        current_date = start_dt + timedelta(days=day_offset)
        day_of_week = current_date.weekday()
        month = current_date.month
        season = get_season(month)
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # Special calendar events / festival spikes
        is_diwali_season = (month == 11 and 1 <= current_date.day <= 5)
        is_new_year = (month == 12 and current_date.day >= 28) or (month == 1 and current_date.day <= 2)
        is_holi_surge = (month == 3 and 22 <= current_date.day <= 26)

        for route_id, info in ROUTE_PROFILES.items():
            # For each route, schedule several departures per day
            # Major routes get 5-6 daily departures, smaller get 3-4
            if info["profile"] in ["commercial_express", "tourist_hill", "tourist_commercial"]:
                daily_hours = [7, 9, 12, 17, 19, 22]
            else:
                daily_hours = [8, 10, 14, 18]

            for hour in daily_hours:
                # Add slight minute variation (e.g. 00, 15, 30, 45)
                minute = int(np.random.choice([0, 15, 30, 45]))
                departure_dt = current_date.replace(hour=hour, minute=minute, second=0)

                is_peak_hour = 1 if hour in [7, 8, 9, 10, 17, 18, 19, 20] else 0
                bus_capacity = int(np.random.choice(info["common_capacities"]))

                # Multipliers
                m_hour = calculate_hourly_multiplier(hour, info["profile"])
                m_day = calculate_weekday_multiplier(day_of_week, info["profile"])
                m_season = calculate_seasonal_multiplier(month, season, info["profile"])

                event_boost = 1.0
                if is_diwali_season:
                    event_boost = 1.25
                elif is_new_year and info["profile"] == "tourist_hill":
                    event_boost = 1.35
                elif is_holi_surge:
                    event_boost = 1.20

                # Expected base passenger demand
                expected_demand = (
                    info["base_demand"]
                    * m_hour
                    * m_day
                    * m_season
                    * event_boost
                )

                # Add realistic stochastic variation (Gaussian noise)
                noise = np.random.normal(0, info["demand_std"])
                raw_passenger_count = expected_demand + noise

                # Realistic bound: passenger count cannot be negative
                # Realistic upper bound: passenger count can exceed bus_capacity during overflow
                # (e.g. up to 125% of capacity representing unaccommodated waitlist demand)
                passenger_count = max(0, int(np.round(raw_passenger_count)))

                records.append({
                    "trip_id": f"TRIP_{trip_counter:06d}",
                    "route_id": route_id,
                    "origin": info["origin"],
                    "destination": info["destination"],
                    "departure_datetime": departure_dt,
                    "departure_hour": hour,
                    "day_of_week": day_of_week,
                    "month": month,
                    "season": season,
                    "is_weekend": is_weekend,
                    "is_peak_hour": is_peak_hour,
                    "bus_capacity": bus_capacity,
                    "passenger_count": passenger_count,
                })
                trip_counter += 1

    df = pd.DataFrame(records)

    # Sort strictly chronologically
    df = df.sort_values(by=["departure_datetime", "route_id"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Leakage-Safe Historical Route Demand Generation:
    # Prediction cutoff: Exactly 24 hours prior to departure_datetime.
    # For trip i at departure_datetime T:
    # Only trips for the same route that departed BEFORE (T - 24 hours) may be used.
    # We compute the rolling 14-day historical mean passenger demand for that route.
    # -------------------------------------------------------------------------
    print("Computing leakage-safe historical route demand (24h cutoff)...")
    df["historical_route_demand"] = np.nan

    # Group by route and compute past 14-day rolling mean ending at T - 24h
    historical_demand_series = []
    
    # We index by departure_datetime for efficient time-window slicing
    for route_id in ROUTE_PROFILES.keys():
        route_mask = (df["route_id"] == route_id)
        route_df = df[route_mask].copy()
        
        times = route_df["departure_datetime"].values
        counts = route_df["passenger_count"].values
        hist_values = np.zeros(len(route_df), dtype=float)

        base_val = ROUTE_PROFILES[route_id]["base_demand"]

        # Vectorized / sorted search for cutoff
        # Cutoff = current_trip_time - 24 hours
        cutoff_deltas = np.timedelta64(24, 'h')
        lookback_window = np.timedelta64(14, 'D')

        for i in range(len(route_df)):
            t_curr = times[i]
            t_cutoff = t_curr - cutoff_deltas
            t_window_start = t_cutoff - lookback_window

            # Find past completed trips within [t_window_start, t_cutoff]
            # Since array is chronologically sorted, we search valid indices
            valid_mask = (times < t_cutoff) & (times >= t_window_start)
            if np.any(valid_mask):
                hist_values[i] = np.mean(counts[valid_mask])
            else:
                # Early warm-up period fallback (use prior baseline estimate)
                hist_values[i] = base_val

        df.loc[route_mask, "historical_route_demand"] = np.round(hist_values, 2)

    # Reorder columns as specified in prompt
    column_order = [
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
    df = df[column_order]

    # Convert departure_datetime to string ISO format for CSV persistence
    df["departure_datetime"] = df["departure_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Generated {len(df):,} synthetic trip records across {len(ROUTE_PROFILES)} routes.")
    print(f"Date range: {df['departure_datetime'].min()} to {df['departure_datetime'].max()}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset successfully saved to: {output_path}")

    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    out_file = os.path.join(out_dir, "synthetic_bus_trips.csv")
    generate_synthetic_data(
        start_date="2024-01-01",
        end_date="2024-12-31",
        random_seed=42,
        output_path=out_file,
    )
