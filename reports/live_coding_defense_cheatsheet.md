# 10-Minute Live Coding & Debugging Task Cheat Sheet
## Algions AI Labs Technical Interview Preparation Guide

This cheat sheet provides exact file locations, code snippets, and mental models for the **10-minute live change/debugging task** during your technical interview.

---

### Scenario 1: "Add a new bus route / corridor"
**Objective:** Add a new route (e.g., `DEL-AGR`: Delhi to Agra, or `JAI-UDP`: Jaipur to Udaipur).

1. **Open:** [`src/data_generator.py`](../src/data_generator.py)
   Add the corridor profile to `ROUTE_PROFILES`:
   ```python
   "DEL-AGR": {
       "origin": "Delhi",
       "destination": "Agra",
       "base_demand": 36.0,
       "demand_std": 4.5,
       "profile": "tourist_commercial",
       "distance_km": 210,
       "common_capacities": [45, 50],
   },
   ```
2. **If adding to live inference directly:**
   In [`src/predict.py`](../src/predict.py), the model artifact stores `route_stats`. For cold-start routes not present in training, `predict.py` safely looks up metadata or you can add a fallback prior:
   ```python
   # In validate_inference_inputs or predict_demand:
   historical_demand = route_info.get("historical_mean", 35.0)
   ```
3. **Verify:** Run a test inference:
   ```bash
   python -m src.predict --route DEL-AGR --departure "2025-07-10 09:00:00" --capacity 45
   ```

---

### Scenario 2: "Modify the capacity alert thresholds or recommendations"
**Objective:** Change overcapacity alert to trigger at 90% occupancy instead of 100%, or change underutilization threshold to 60%.

1. **Open:** [`src/predict.py`](../src/predict.py) around line 130 (`predict_demand`):
   ```python
   # Current thresholds:
   # if occupancy_pct > 100.0: ...
   # elif occupancy_pct >= 85.0: ...
   
   # LIVE CHANGE: e.g. Trigger alert at 90%
   if occupancy_pct >= 90.0:
       demand_category = "Overcapacity Warning"
       alert_level = "CRITICAL_OVERFLOW"
       capacity_alert = f"Occupancy exceeds 90% threshold ({occupancy_pct}% load)!"
       capacity_action = "Stage relief bus at depot."
   elif occupancy_pct < 60.0:
       demand_category = "Underutilization Warning"
       alert_level = "UNDERUTILIZED"
   ```
2. **Update Streamlit UI badge styling (if requested):**
   In [`app/streamlit_app.py`](../app/streamlit_app.py), under `🔮 Demand Prediction` tab, the badge dynamically uses `result['alert_level']`.
3. **Test:** Run pytest:
   ```bash
   python -m pytest tests/test_inference.py -v
   ```

---

### Scenario 3: "Add a new feature (e.g. `is_holiday` or `temperature`)"
**Objective:** Add a binary feature `is_holiday` to the preprocessing pipeline.

1. **Step 1: Feature definition in [`src/preprocessing.py`](../src/preprocessing.py):**
   Add `"is_holiday"` to `FEATURE_COLUMNS`:
   ```python
   FEATURE_COLUMNS = [
       "route_id", "origin", "destination", "departure_hour",
       "day_of_week", "month", "season", "is_weekend", "is_peak_hour",
       "is_holiday",  # <-- NEW FEATURE
       "bus_capacity", "historical_route_demand",
   ]
   ```
2. **Step 2: Pipeline preprocessor in [`src/feature_engineering.py`](../src/feature_engineering.py):**
   Add `"is_holiday"` to `numeric_cols` in `create_feature_pipeline()`:
   ```python
   numeric_cols = [
       ..., "is_peak_hour", "is_holiday", "bus_capacity", ...
   ]
   ```
3. **Step 3: Inference calculation in [`src/predict.py`](../src/predict.py):**
   Derive the feature at runtime:
   ```python
   # Example holiday dates (Independence Day, Republic Day, Gandhi Jayanti, Diwali)
   holidays = [(1, 26), (8, 15), (10, 2), (11, 1)]
   is_holiday = 1 if (dep_dt.month, dep_dt.day) in holidays else 0
   ```
   Add `"is_holiday": is_holiday` into `input_data`.

---

### Scenario 4: "Switch the candidate model or change hyperparameter search space"
**Objective:** Adjust tree depth or test another estimator (e.g., `GradientBoostingRegressor` or `LinearRegression`).

1. **Open:** [`src/train.py`](../src/train.py)
   Locate candidate tuning blocks:
   ```python
   # To add deeper trees or adjust learning rates:
   hgb_candidates = [
       {"max_iter": 150, "learning_rate": 0.05, "max_depth": 10, "min_samples_leaf": 15, "l2_regularization": 0.5},
       {"max_iter": 250, "learning_rate": 0.03, "max_depth": 12, "min_samples_leaf": 10, "l2_regularization": 0.8}, # New
   ]
   ```
2. **Retrain:**
   ```bash
   python -m src.train
   ```
   The script automatically logs validation metrics, selects the champion with the lowest WAPE, and saves the updated pipeline to `models/best_model.joblib`.

---

### Scenario 5: "Add a new unit test in Pytest"
**Objective:** Write a test verifying that night departures (22:00–05:00) never throw errors.

1. **Open:** [`tests/test_inference.py`](../tests/test_inference.py)
2. **Add:**
   ```python
   def test_night_departure_inference(model_exists):
       """Verify inference works seamlessly for late night departures."""
       res = predict_demand(
           route_id="DEL-CHD",
           departure_datetime="2025-08-14 23:30:00",
           bus_capacity=45,
       )
       assert res["adjusted_prediction"] >= 0
       assert "Night" in res["explanation_factors"][0] or "off-peak" in res["explanation_factors"][0].lower()
   ```
3. **Execute:**
   ```bash
   python -m pytest tests/test_inference.py -v
   ```

---

### Scenario 6: "Run inference from CLI or batch script"
```bash
# Direct CLI test:
python -m src.predict --route DEL-CHD --departure "2025-06-20 08:30:00" --capacity 45

# Or inside Python interactive shell:
from src.predict import predict_demand
res = predict_demand("DEL-JAI", "2025-11-15 18:00:00", bus_capacity=50)
print(f"Predicted Pax: {res['adjusted_prediction']}, Occupancy: {res['occupancy_percentage']}%")
```

---

### Core Formula Memory Anchor
- **WAPE:** $\frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$ (Volume-weighted, robust against zero division).
- **Occupancy %:** $\frac{\text{Adjusted Demand}}{\text{Bus Capacity}} \times 100\%$ (Uncapped, reveals spillover).
- **Historical Demand Cutoff:** Departures prior to $T - 24\text{ hours}$ (Strict leakage prevention).
- **Partitioning:** 70% Train (Jan–Sep), 15% Validation (Sep–Nov), 15% Test (Nov–Dec).

