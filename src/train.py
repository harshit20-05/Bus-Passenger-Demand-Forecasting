"""
Model Training and Model Selection Pipeline
===========================================
Trains baseline, RandomForestRegressor, and HistGradientBoostingRegressor.
Tunes hyperparameters, evaluates on validation split, selects champion model,
and exports full end-to-end Joblib pipeline artifact with metadata.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    load_dataset,
    validate_and_clean_data,
    chronological_split,
    prepare_features_and_targets,
    FEATURE_COLUMNS,
)
from src.feature_engineering import create_feature_pipeline
from src.baseline import HistoricalTimeSlotBaseline, calculate_metrics


def train_and_select_best_model(
    data_path: str = "data/raw/synthetic_bus_trips.csv",
    models_dir: str = "models",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes end-to-end training, hyperparameter comparison, and artifact saving.
    """
    print(f"Loading raw data from: {data_path}")
    raw_df = load_dataset(data_path)

    print("Validating and cleaning dataset...")
    cleaned_df, quality_report = validate_and_clean_data(raw_df)

    print("Executing strict chronological split (70% Train, 15% Val, 15% Test)...")
    train_df, val_df, test_df = chronological_split(cleaned_df, 0.70, 0.15, 0.15)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_targets(
        train_df, val_df, test_df
    )

    # ---------------------------------------------------------
    # 1. Historical Baseline Model
    # ---------------------------------------------------------
    print("\n--- Fitting Baseline Model ---")
    baseline_model = HistoricalTimeSlotBaseline()
    baseline_model.fit(X_train, y_train)

    val_preds_baseline = baseline_model.predict(X_val)
    baseline_val_metrics = calculate_metrics(y_val.values, val_preds_baseline)
    print(f"Baseline Validation Metrics: {baseline_val_metrics}")

    # ---------------------------------------------------------
    # 2. Build Preprocessor Pipeline
    # ---------------------------------------------------------
    print("\n--- Initializing Preprocessing & Feature Engineering Pipeline ---")
    preprocessor, cat_cols, num_cols = create_feature_pipeline()

    # Pre-transform training and validation features for fast tuning
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    # ---------------------------------------------------------
    # 3. Model 1: RandomForestRegressor
    # ---------------------------------------------------------
    print("\n--- Training Model 1: RandomForestRegressor ---")
    # Grid search / hyperparameter comparison on validation set
    rf_candidates = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 150, "max_depth": 15, "min_samples_split": 4},
        {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2},
    ]

    best_rf_model = None
    best_rf_metrics = {"WAPE": float("inf"), "MAE": float("inf"), "RMSE": float("inf")}
    best_rf_params = None

    for params in rf_candidates:
        rf = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=random_seed,
            n_jobs=-1,
        )
        rf.fit(X_train_transformed, y_train)
        val_preds = rf.predict(X_val_transformed)
        metrics = calculate_metrics(y_val.values, val_preds)
        print(f"  RF (params={params}) -> Val MAE: {metrics['MAE']}, WAPE: {metrics['WAPE']}%")

        if metrics["WAPE"] < best_rf_metrics["WAPE"]:
            best_rf_metrics = metrics
            best_rf_model = rf
            best_rf_params = params

    print(f"Best RF Params: {best_rf_params} | Val Metrics: {best_rf_metrics}")

    # ---------------------------------------------------------
    # 4. Model 2: HistGradientBoostingRegressor
    # ---------------------------------------------------------
    print("\n--- Training Model 2: HistGradientBoostingRegressor ---")
    hgb_candidates = [
        {"max_iter": 100, "learning_rate": 0.08, "max_depth": 8, "min_samples_leaf": 20, "l2_regularization": 0.1},
        {"max_iter": 150, "learning_rate": 0.05, "max_depth": 10, "min_samples_leaf": 15, "l2_regularization": 0.5},
        {"max_iter": 200, "learning_rate": 0.06, "max_depth": 12, "min_samples_leaf": 10, "l2_regularization": 0.2},
    ]

    best_hgb_model = None
    best_hgb_metrics = {"WAPE": float("inf"), "MAE": float("inf"), "RMSE": float("inf")}
    best_hgb_params = None

    for params in hgb_candidates:
        hgb = HistGradientBoostingRegressor(
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            random_state=random_seed,
        )
        hgb.fit(X_train_transformed, y_train)
        val_preds = hgb.predict(X_val_transformed)
        metrics = calculate_metrics(y_val.values, val_preds)
        print(f"  HGB (params={params}) -> Val MAE: {metrics['MAE']}, WAPE: {metrics['WAPE']}%")

        if metrics["WAPE"] < best_hgb_metrics["WAPE"]:
            best_hgb_metrics = metrics
            best_hgb_model = hgb
            best_hgb_params = params

    print(f"Best HGB Params: {best_hgb_params} | Val Metrics: {best_hgb_metrics}")

    # ---------------------------------------------------------
    # 5. Validation Model Selection
    # ---------------------------------------------------------
    print("\n--- Validation Performance Comparison ---")
    print(f"Baseline                  : WAPE = {baseline_val_metrics['WAPE']}% | MAE = {baseline_val_metrics['MAE']}")
    print(f"RandomForestRegressor     : WAPE = {best_rf_metrics['WAPE']}% | MAE = {best_rf_metrics['MAE']}")
    print(f"HistGradientBoostingRegr  : WAPE = {best_hgb_metrics['WAPE']}% | MAE = {best_hgb_metrics['MAE']}")

    # Select champion based on lowest validation WAPE
    if best_hgb_metrics["WAPE"] <= best_rf_metrics["WAPE"]:
        champion_name = "HistGradientBoostingRegressor"
        champion_estimator = best_hgb_model
        champion_params = best_hgb_params
        champion_val_metrics = best_hgb_metrics
    else:
        champion_name = "RandomForestRegressor"
        champion_estimator = best_rf_model
        champion_params = best_rf_params
        champion_val_metrics = best_rf_metrics

    print(f"\n Champion Model Selected: {champion_name}")

    # ---------------------------------------------------------
    # 6. Build Final Production Pipeline
    # ---------------------------------------------------------
    # Encapsulate preprocessing and champion estimator into a single Pipeline
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", champion_estimator),
        ]
    )

    # Compute historical demand lookup statistics per route for inference
    route_stats = (
        train_df.groupby("route_id")
        .agg(
            historical_mean=("passenger_count", "mean"),
            historical_median=("passenger_count", "median"),
            origin=("origin", "first"),
            destination=("destination", "first"),
            common_capacity=("bus_capacity", lambda s: int(s.mode()[0])),
        )
        .to_dict(orient="index")
    )

    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "best_model.joblib")

    # Metadata bundle
    artifact = {
        "pipeline": final_pipeline,
        "champion_model_name": champion_name,
        "champion_params": champion_params,
        "validation_metrics": {
            "baseline": baseline_val_metrics,
            "random_forest": best_rf_metrics,
            "hist_gradient_boosting": best_hgb_metrics,
            "champion": champion_val_metrics,
        },
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "route_stats": route_stats,
        "data_quality_report": quality_report,
        "training_date_range": (
            str(train_df["departure_datetime"].min()),
            str(train_df["departure_datetime"].max()),
        ),
        "validation_date_range": (
            str(val_df["departure_datetime"].min()),
            str(val_df["departure_datetime"].max()),
        ),
        "test_date_range": (
            str(test_df["departure_datetime"].min()),
            str(test_df["departure_datetime"].max()),
        ),
    }

    joblib.dump(artifact, model_save_path)
    print(f"Model and Preprocessing Pipeline saved to: {model_save_path}")

    # Also save metadata as a readable JSON
    meta_json_path = os.path.join(models_dir, "model_metadata.json")
    json_safe_metadata = {k: v for k, v in artifact.items() if k != "pipeline"}
    with open(meta_json_path, "w") as f:
        json.dump(json_safe_metadata, f, indent=2, default=str)
    print(f"Model metadata saved to: {meta_json_path}")

    return artifact


if __name__ == "__main__":
    train_and_select_best_model()

