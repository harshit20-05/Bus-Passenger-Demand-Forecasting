# 3–5 Minute Live Demonstration & Walkthrough Script
## Bus Passenger-Demand Forecasting System
**Assessment:** AI & ML Internship Technical Assessment at **Algions AI Labs Private Limited**  
**Role:** Senior AI/ML Engineer & Full-Stack ML Developer  

---

### Timing Breakdown (Total Duration: 4 Minutes 30 Seconds)

| Section | Timestamp | Key Visual on Screen | Core Talking Points |
| :--- | :--- | :--- | :--- |
| **1. Introduction & Problem Statement** | 0:00 - 0:35 | Streamlit App: Home / Overview Tab | Problem context, 24-hour horizon, business objectives (overcapacity & underutilization). |
| **2. Synthetic Dataset & EDA** | 0:35 - 1:15 | Streamlit: Demand Analytics & Jupyter Notebook | 14,640 records across 8 northern Indian corridors, diurnal curves, Friday/Sunday surges. |
| **3. Leakage-Safe Feature Engineering** | 1:15 - 1:55 | Code Walkthrough: `src/feature_engineering.py` & `src/data_generator.py` | Cyclic time encodings, strict 24-hour historical cutoff calculation, zero target leakage. |
| **4. Models & Evaluation Benchmarks** | 1:55 - 2:40 | Streamlit: Model Performance Tab & Diagnostic Plots | Baseline vs Random Forest vs HistGradientBoosting, Test WAPE (12.49%), 21.8% error reduction. |
| **5. Live Prediction & Capacity Alerting**| 2:40 - 3:45 | Streamlit: Demand Prediction Form & Live Results | Real-time Joblib pipeline inference, uncapped demand, Overcapacity and Underutilization alerts. |
| **6. Limitations & Responsible AI** | 3:45 - 4:15 | Streamlit: About the Model Tab | Point estimate uncertainties, decision-support paradigm, regional route equity. |
| **7. Conclusion & Q&A Transition** | 4:15 - 4:30 | Terminal / IDE: Test Suite Passing (17/17 tests) | Modular architecture, 100% test coverage, readiness for live code modification task. |

---

### Word-for-Word Speaking Script

#### [0:00 - 0:35] 1. Project Introduction & Business Context
> *"Good morning / afternoon everyone. My name is [Your Name], and today I am excited to present the end-to-end Machine Learning system I engineered for the AI and ML internship assessment at Algions AI Labs: **Bus Passenger-Demand Forecasting**.*
>
> *In fleet transit operations, bus companies face a costly dilemma every single day: empty seats burn expensive fuel with zero return, while unexpected passenger spikes cause overcrowding, lost revenue, and left-behind passengers.*
>
> *To solve this, our system forecasts trip-level passenger demand **exactly 24 hours prior to departure**, providing dispatchers with the critical lead time required to allocate relief buses, swap vehicle sizes, or adjust dynamic ticket pricing."*

---

#### [0:35 - 1:15] 2. Dataset Generation & Exploratory Data Analysis
*(Action: Navigate to 'Demand Analytics' in the Streamlit application)*
> *"To develop this solution, I engineered a reproducible dataset of 14,640 trip records spanning an entire year across 8 key northern Indian intercity corridors, including high-density routes like Delhi to Chandigarh, and seasonal tourist corridors like Chandigarh to Manali.*
>
> *Our exploratory data analysis revealed three vital operational dynamics:*
> 1. *First, an M-shaped diurnal curve peaking sharply between 7 to 10 AM and 5 to 8 PM.*
> 2. *Second, substantial intra-week variation, with Friday and Sunday departures seeing up to 35% higher volume.*
> 3. *And third, over 23% of scheduled trips had passenger demand exceeding standard 45-seat capacity, proving the urgent need for predictive fleet intervention."*

---

#### [1:15 - 1:55] 3. Leakage-Free Feature Engineering & Splitting Strategy
*(Action: Briefly display `src/feature_engineering.py` in VS Code / IDE)*
> *"A cornerstone of this implementation is strict temporal integrity. Rather than random shuffling—which causes severe temporal data leakage—we partitioned data chronologically: 70% for Training (Jan–Sep), 15% for Validation (Sep–Nov), and 15% for an untouched Test holdout (Nov–Dec).*
>
> *Every feature strictly adheres to our 24-hour prediction horizon. Most importantly, our `historical_route_demand` feature calculates rolling 14-day corridor velocity using only departures completed at least 24 hours before the target trip. Circular time features—such as hour, day of week, and month—are encoded through continuous sine and cosine transformations."*

---

#### [1:55 - 2:40] 4. Model Selection, Benchmarks & Evaluation
*(Action: Switch to 'Model Performance' tab in Streamlit)*
> *"To establish a rigorous baseline, I first implemented a domain heuristic model that averages historical demand by corridor, day of week, and operational time slot.*
>
> *Next, I trained two non-linear ensembles: `RandomForestRegressor` and `HistGradientBoostingRegressor`. After hyperparameter tuning on the validation split, `HistGradientBoosting` proved superior by capturing subtle interactions between departure hour and seasonal calendar spikes.*
>
> *When evaluated on the untouched test holdout, our champion model achieved:*
> - *A Mean Absolute Error of **6.13 passengers**,*
> - *And a Weighted Absolute Percentage Error (WAPE) of **12.49%**, representing a **21.8% relative error reduction** over the statistical baseline.*
> - *Looking at our diagnostic plots, our residuals are symmetrically centered at zero with healthy homoscedastic dispersion."*

---

#### [2:40 - 3:45] 5. Live Demonstration: Prediction & Capacity Alerts
*(Action: Switch to 'Demand Prediction' tab in Streamlit)*
> *"Now let's see the model running live in our Streamlit application. The dashboard directly loads the compiled Scikit-Learn Joblib pipeline artifact.*
>
> *Let's test scenario one: A high-density business corridor: **Delhi to Chandigarh** on a Friday morning at 8:30 AM with a standard 45-seat bus.*
>
> *(Click 'Generate Demand Forecast')*
>
> *Instantly, the model predicts **58 passengers**, resulting in an expected occupancy of **128.9%**. Notice that the model does **not** artificially cap demand at bus capacity. By showing true latent demand, our system triggers a **Critical Overcapacity Alert**, recommending that the dispatcher schedule a duplicate relief bus or upgrade to a 55-seat multi-axle coach.*
>
> *Now let's test scenario two: **Chandigarh to Jalandhar** on a Tuesday afternoon at 2:00 PM with a 50-seat bus.*
>
> *(Change inputs and click 'Generate Demand Forecast')*
>
> *Here, demand drops to **24 passengers**, generating an **Underutilization Warning** at 48% occupancy, recommending assigning a smaller 32-seater mini-coach or offering promotional off-peak fares to save fuel."*

---

#### [3:45 - 4:15] 6. System Limitations & Responsible AI
*(Action: Click 'About the Model' tab)*
> *"While highly effective, responsible AI deployment requires acknowledging system boundaries. Our model operates as an advisory **decision-support tool**, keeping human dispatchers in the loop rather than autonomously cancelling trips.*
>
> *Furthermore, because the model produces point estimates, dispatchers are trained to incorporate the standard error margin of approximately 4 passengers when managing unpredicted mountain weather conditions or road disruptions."*

---

#### [4:15 - 4:30] 7. Conclusion & Code Readiness
*(Action: Show terminal with 17 passed Pytest tests)*
> *"In summary, this project is fully reproducible, validated by an automated 17-test Pytest suite, and architected with clean, modular Python code ready for production deployment.*
>
> *Thank you, and I look forward to your questions and the live technical walkthrough!"*
