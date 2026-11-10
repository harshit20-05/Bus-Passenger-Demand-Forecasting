"""
Model Evaluation and Diagnostics Module
=======================================
Evaluates trained models on the untouched test dataset.
Generates:
1. Candidate model comparison table (Baseline, Random Forest, HistGradientBoosting)
2. Route-wise error breakdown (MAE, RMSE, WAPE)
3. Peak vs Off-Peak error breakdown
4. Residual analysis and distribution
5. Actual vs Predicted visualizations
6. Feature importance (Permutation Importance on test set)
7. Identification of strongest and weakest routes
All figures saved to reports/figures/ and metrics exported to reports/evaluation_results.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from typing import Dict, Any, Tuple

from src.preprocessing import (
    load_dataset,
    validate_and_clean_data,
    chronological_split,
    prepare_features_and_targets,
)
from src.baseline import HistoricalTimeSlotBaseline, calculate_metrics


# Set clean aesthetic styling for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def evaluate_models_on_test(
    data_path: str = "data/raw/synthetic_bus_trips.csv",
    model_path: str = "models/best_model.joblib",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Executes comprehensive evaluation on the untouched test split.
    """
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading artifact from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train model first!")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    champion_name = artifact["champion_model_name"]

    print(f"Loading data from: {data_path}")
    raw_df = load_dataset(data_path)
    cleaned_df, _ = validate_and_clean_data(raw_df)
    train_df, val_df, test_df = chronological_split(cleaned_df, 0.70, 0.15, 0.15)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_targets(
        train_df, val_df, test_df
    )

    # -------------------------------------------------------------
    # 1. Baseline Test Evaluation
    # -------------------------------------------------------------
    baseline = HistoricalTimeSlotBaseline()
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    baseline_metrics = calculate_metrics(y_test.values, y_pred_baseline)

    # -------------------------------------------------------------
    # 2. Champion Model Test Prediction
    # -------------------------------------------------------------
    y_pred_champion_raw = pipeline.predict(X_test)
    # Business-facing non-negative adjustment (passenger count cannot be negative)
    y_pred_champion = np.maximum(0.0, y_pred_champion_raw)
    champion_metrics = calculate_metrics(y_test.values, y_pred_champion)

    print(f"Test Set Metrics - Baseline: {baseline_metrics}")
    print(f"Test Set Metrics - Champion ({champion_name}): {champion_metrics}")

    # Build evaluation DataFrame for test set
    eval_df = test_df.copy()
    eval_df["predicted_demand"] = y_pred_champion
    eval_df["residual"] = eval_df["passenger_count"] - eval_df["predicted_demand"]
    eval_df["abs_error"] = np.abs(eval_df["residual"])

    # -------------------------------------------------------------
    # 3. Route-Wise Breakdown
    # -------------------------------------------------------------
    route_metrics = {}
    for route, grp in eval_df.groupby("route_id"):
        m = calculate_metrics(grp["passenger_count"].values, grp["predicted_demand"].values)
        m["trip_count"] = int(len(grp))
        m["mean_actual"] = round(float(grp["passenger_count"].mean()), 2)
        route_metrics[route] = m

    route_df = pd.DataFrame(route_metrics).T.sort_values(by="WAPE")
    strongest_route = route_df.index[0]
    weakest_route = route_df.index[-1]

    # -------------------------------------------------------------
    # 4. Peak vs Off-Peak Breakdown
    # -------------------------------------------------------------
    peak_metrics = {}
    for is_peak, grp in eval_df.groupby("is_peak_hour"):
        label = "Peak_Hours" if is_peak == 1 else "Off_Peak_Hours"
        m = calculate_metrics(grp["passenger_count"].values, grp["predicted_demand"].values)
        m["trip_count"] = int(len(grp))
        m["mean_actual"] = round(float(grp["passenger_count"].mean()), 2)
        peak_metrics[label] = m

    # -------------------------------------------------------------
    # 5. Visualizations Generation
    # -------------------------------------------------------------
    print("Generating evaluation figures...")

    # Plot 1: Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(eval_df["passenger_count"], eval_df["predicted_demand"], alpha=0.35, color="#1f77b4", edgecolors="none")
    max_val = max(eval_df["passenger_count"].max(), eval_df["predicted_demand"].max()) + 5
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Ideal 1:1 Reference Line")
    ax.set_title(f"Actual vs Predicted Passenger Demand (Test Set: {champion_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Passenger Count", fontsize=11)
    ax.set_ylabel("Predicted Passenger Count", fontsize=11)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig1_path = os.path.join(figures_dir, "actual_vs_predicted.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # Plot 2: Residual Analysis (Histogram & Residuals vs Predicted)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(eval_df["residual"], kde=True, ax=axes[0], color="#2ca02c", bins=30)
    axes[0].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_title("Residual Distribution (Actual - Predicted)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Residual (Passengers)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)

    axes[1].scatter(eval_df["predicted_demand"], eval_df["residual"], alpha=0.35, color="#ff7f0e", edgecolors="none")
    axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title("Residual vs Predicted Value (Homoscedasticity Check)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Passenger Demand", fontsize=11)
    axes[1].set_ylabel("Residual (Passengers)", fontsize=11)
    fig.tight_layout()
    fig2_path = os.path.join(figures_dir, "residual_analysis.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)

    # Plot 3: Route-wise WAPE comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        x=route_df.index,
        y=route_df["WAPE"],
        hue=route_df.index,
        ax=ax,
        palette="Blues_r",
        legend=False,
    )
    ax.set_title("Forecasting Error (WAPE %) Across Bus Corridors", fontsize=13, fontweight="bold")
    ax.set_xlabel("Route ID", fontsize=11)
    ax.set_ylabel("WAPE (%)", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig3_path = os.path.join(figures_dir, "route_wise_error.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)

    # Plot 4: Peak vs Off-Peak Error comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    peak_df_plot = pd.DataFrame(peak_metrics).T
    sns.barplot(
        x=peak_df_plot.index,
        y=peak_df_plot["WAPE"],
        hue=peak_df_plot.index,
        ax=ax,
        palette=["#d62728", "#1f77b4"],
        legend=False,
    )
    ax.set_title("WAPE (%) Comparison: Peak vs Off-Peak Departure Slots", fontsize=12, fontweight="bold")
    ax.set_xlabel("Operational Period", fontsize=11)
    ax.set_ylabel("WAPE (%)", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.2),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig4_path = os.path.join(figures_dir, "peak_vs_offpeak_error.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)

    # Plot 5: Feature Importance via Permutation Importance on Test Set
    print("Calculating permutation feature importance...")
    perm_result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    feature_names = X_test.columns.tolist()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values(by="importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(
        importance_df["feature"],
        importance_df["importance_mean"],
        xerr=importance_df["importance_std"],
        color="#2563EB",
        edgecolor="#1E40AF",
        capsize=4,
    )
    ax.set_title(f"Permutation Feature Importance on Test Set ({champion_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Score (R² / Explanatory Power)", fontsize=11)
    ax.set_ylabel("Raw Input Feature", fontsize=11)
    fig.tight_layout()
    fig5_path = os.path.join(figures_dir, "feature_importance.png")
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)

    # Plot 6: 7-Day Sample Chronological Forecast vs Actual
    sample_window = eval_df[eval_df["route_id"] == "DEL-CHD"].head(30)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(range(len(sample_window)), sample_window["passenger_count"], "o-", label="Actual Demand", color="#1f77b4", lw=2)
    ax.plot(range(len(sample_window)), sample_window["predicted_demand"], "s--", label="Model Forecast", color="#d62728", lw=2)
    ax.set_title("Sample Forecast Horizon (Delhi - Chandigarh Corridor)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sequential Test Trip Index", fontsize=11)
    ax.set_ylabel("Passenger Count", fontsize=11)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig6_path = os.path.join(figures_dir, "sample_forecast_timeline.png")
    fig.savefig(fig6_path, dpi=300)
    plt.close(fig)

    # Compile evaluation results dictionary
    evaluation_results = {
        "champion_model_name": champion_name,
        "test_metrics": {
            "baseline": baseline_metrics,
            "champion": champion_metrics,
        },
        "route_wise_metrics": route_metrics,
        "peak_vs_offpeak_metrics": peak_metrics,
        "strongest_route": {
            "route_id": strongest_route,
            "metrics": route_metrics[strongest_route],
        },
        "weakest_route": {
            "route_id": weakest_route,
            "metrics": route_metrics[weakest_route],
        },
        "feature_importances": importance_df.to_dict(orient="records"),
        "test_sample_size": len(test_df),
    }

    # Save to JSON
    json_path = os.path.join(reports_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"Evaluation results saved to: {json_path}")

    return evaluation_results


if __name__ == "__main__":
    evaluate_models_on_test()
