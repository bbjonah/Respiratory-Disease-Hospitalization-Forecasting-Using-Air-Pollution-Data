# Respiratory Disease Hospitalization Forecasting Using Air Pollution Data

> **Environmental Health Analytics • Machine Learning • Time-Series Forecasting • Public Health Intelligence**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Respiratory diseases place substantial pressure on healthcare systems, particularly when environmental conditions contribute to increased respiratory illness and hospital demand.

This project develops an **environmental health forecasting framework** for predicting respiratory disease hospitalizations using a combination of:

* Air pollution indicators
* Weather conditions
* Demographic risk factors
* Historical hospitalization patterns
* Temporal and seasonal features

A **Random Forest Regression** model is used to estimate hospitalization demand while time-series feature engineering captures recent hospitalization trends and recurring temporal patterns.

The broader objective is to demonstrate how environmental and healthcare data can be integrated into a machine-learning workflow capable of supporting **healthcare demand forecasting, environmental health surveillance, hospital preparedness, and public-health planning**.

---

## Problem Statement

Poor air quality is an important environmental health concern and has been associated with adverse respiratory outcomes.

Healthcare systems may experience increased demand during periods characterized by unfavorable environmental conditions. Being able to anticipate potential changes in respiratory hospitalization demand could help public-health and healthcare stakeholders prepare resources before demand increases.

This project investigates the following question:

> **Can environmental, demographic, temporal, and historical hospitalization indicators be used to forecast respiratory disease hospitalizations?**

---

## Project Objectives

The project is designed to:

1. Analyze relationships between air pollution and respiratory hospitalizations.
2. Examine temporal and seasonal hospitalization patterns.
3. Engineer historical and rolling hospitalization features for forecasting.
4. Develop a machine-learning regression model for hospitalization prediction.
5. Evaluate forecasting performance using standard regression metrics.
6. Identify the most influential predictive features.
7. Generate short-term hospitalization forecasts.
8. Demonstrate potential applications in public-health preparedness and healthcare resource planning.

---

## Analytical Framework

The project follows an end-to-end data science workflow:

```text
Environmental & Demographic Data
              │
              ▼
        Data Validation
              │
              ▼
     Exploratory Data Analysis
              │
              ▼
      Temporal Feature Engineering
              │
              ▼
       Time-Based Data Splitting
              │
              ▼
      Random Forest Regression
              │
              ▼
       Model Evaluation
              │
              ▼
      Feature Importance Analysis
              │
              ▼
       Future Forecasting
              │
              ▼
   Public Health Interpretation
```

---

# Dataset

## Data Source

The project uses a **realistic synthetic dataset** designed to simulate environmental and public-health conditions across multiple geographic regions and time periods.

The dataset was created for analytical demonstration and does **not** represent real patient records or actual hospital surveillance data.

## Dataset Characteristics

The dataset contains:

* Daily observations
* Multiple geographic regions
* Air-quality measurements
* Weather indicators
* Demographic risk factors
* Daily respiratory hospitalization counts

## Variables

| Variable                       | Description                                               |
| ------------------------------ | --------------------------------------------------------- |
| `date`                         | Observation date                                          |
| `state`                        | Geographic region                                         |
| `pm25`                         | Fine particulate matter (PM2.5)                           |
| `pm10`                         | Particulate matter (PM10)                                 |
| `no2`                          | Nitrogen dioxide concentration                            |
| `so2`                          | Sulfur dioxide concentration                              |
| `co`                           | Carbon monoxide concentration                             |
| `ozone`                        | Ozone concentration                                       |
| `aqi`                          | Air Quality Index                                         |
| `temperature_c`                | Daily temperature in Celsius                              |
| `humidity_pct`                 | Relative humidity                                         |
| `rainfall_mm`                  | Daily rainfall                                            |
| `wind_speed_kmh`               | Wind speed                                                |
| `population_density`           | Population density                                        |
| `smoking_prevalence_pct`       | Estimated smoking prevalence                              |
| `elderly_population_pct`       | Percentage of elderly population                          |
| `respiratory_hospitalizations` | Daily respiratory hospitalization count — target variable |

---

# Feature Engineering

Because the project is a forecasting problem, historical hospitalization information is incorporated into the modeling framework.

### Hospitalization lag features

The following lag variables are created:

* `hospitalization_lag_1`
* `hospitalization_lag_7`
* `hospitalization_lag_30`

These represent hospitalization levels from:

* The previous day
* Seven days earlier
* Thirty days earlier

### Rolling features

The project also calculates:

* 7-day rolling hospitalization average
* 30-day rolling hospitalization average

Rolling features are shifted so that the current target value is not included in the historical calculation.

This helps reduce **target leakage** and ensures that forecasting features represent information that would have been available before the prediction date.

### Temporal features

Calendar-based predictors include:

* Year
* Month
* Quarter
* Day of week
* Day of year

These features allow the model to capture recurring temporal and seasonal patterns.

---

# Exploratory Data Analysis

The exploratory analysis investigates the relationships between environmental conditions and respiratory hospitalization demand.

### Key analyses include

* Hospitalization distribution
* Daily hospitalization trends
* Regional hospitalization patterns
* PM2.5 versus hospitalization levels
* AQI versus hospitalization levels
* Environmental correlation analysis
* Monthly hospitalization patterns
* Weekly hospitalization patterns
* Pollution indicator distributions

Example analytical questions include:

> Do higher PM2.5 levels correspond with increased hospitalization demand?

> Which environmental variables have the strongest predictive relationship with hospitalization counts?

> Are there recurring seasonal or weekly hospitalization patterns?

---

# Machine Learning Model

## Random Forest Regressor

The primary predictive model is a **Random Forest Regressor**.

Random Forest was selected because it can model complex nonlinear relationships between environmental variables, demographic characteristics, historical hospitalization patterns, and healthcare demand.

### Model configuration

The implementation uses:

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

The model uses multiple decision trees and aggregates their predictions to produce the final hospitalization forecast.

---

# Train/Test Strategy

Because this is a forecasting task, the project uses a **chronological train/test split** rather than a random split.

Approximately:

* **80%** of observations → training set
* **20%** of observations → testing set

The training observations represent the earlier portion of the time series, while the test set contains later observations.

This approach helps prevent future observations from inadvertently influencing the training process.

---

# Model Evaluation

The forecasting model is evaluated using three regression metrics.

### Mean Absolute Error — MAE

Measures the average absolute difference between observed and predicted hospitalization counts.

Lower values indicate better performance.

### Root Mean Squared Error — RMSE

Measures the square root of the average squared prediction error.

RMSE gives greater weight to larger forecasting errors.

### R² Score

Measures the proportion of variation in the target variable explained by the model.

Values closer to 1 indicate stronger explanatory performance.

The notebook also evaluates model performance at the regional level to identify geographic differences in forecasting accuracy.

---

# Model Interpretation

Model-based feature importance is used to identify variables that contribute most strongly to the Random Forest's predictions.

The analysis considers the importance of:

* Historical hospitalization trends
* PM2.5
* PM10
* NO₂
* SO₂
* CO
* Ozone
* AQI
* Weather conditions
* Demographic indicators
* Temporal variables

> **Important:** Feature importance represents predictive usefulness within the fitted model. It does not establish that a variable causes respiratory hospitalizations.

---

# Forecasting

The project includes a short-term future forecasting workflow.

The forecasting process uses:

1. Recent hospitalization history.
2. Historical lag features.
3. Rolling hospitalization averages.
4. Environmental baseline values.
5. Demographic characteristics.
6. Calendar features.

The forecasting process is implemented recursively so that predicted hospitalization values can be incorporated into subsequent lag and rolling calculations.

### Forecast outputs

Generated forecasts include:

```text
date
state
predicted_hospitalizations
```

The resulting forecast can be used to visualize projected healthcare demand across the forecast horizon.

---

# Visualizations

The project generates several publication-ready analytical visualizations.

### Hospitalization Analytics

* Hospitalization distribution
* Daily hospitalization trends
* Regional hospitalization comparisons
* Monthly hospitalization patterns
* Weekly hospitalization patterns

### Environmental Health Analytics

* PM2.5 versus hospitalization levels
* AQI versus hospitalization levels
* Pollution correlations
* Environmental indicator distributions
* Pollution feature importance

### Forecasting Analytics

* Actual versus predicted hospitalizations
* Forecast error distribution
* Future hospitalization forecasts
* Regional model performance

### Model Interpretation

* Feature importance rankings
* Correlation heatmaps
* Pollution-specific feature importance

Generated figures are stored in:

```text
outputs/figures/
```

---

# Project Structure

```text
Respiratory-Disease-Hospitalization-Forecasting-Using-Air-Pollution-Data/
│
├── data/
│   └── respiratory_disease_hospitalization_forecasting_dataset.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── outputs/
│   ├── forecast_predictions.csv
│   ├── future_forecast.csv
│   ├── feature_importance.csv
│   └── figures/
│
├── src/
│   └── hospitalization_forecasting.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Technologies

## Programming Language

* Python 3.9+

## Data Analysis

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn

---

# Installation

Clone the repository:

```bash
git clone https://github.com/bbjonah/Respiratory-Disease-Hospitalization-Forecasting-Using-Air-Pollution-Data.git
```

Navigate to the project directory:

```bash
cd Respiratory-Disease-Hospitalization-Forecasting-Using-Air-Pollution-Data
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

## Run the Python Pipeline

```bash
python src/hospitalization_forecasting.py
```

## Run the Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebooks/analysis.ipynb
```

Run the notebook cells sequentially to reproduce the exploratory analysis, feature engineering, model training, evaluation, feature importance analysis, and forecasting workflow.

---

# Outputs

The project produces the following analytical outputs:

```text
outputs/
│
├── forecast_predictions.csv
├── future_forecast.csv
├── feature_importance.csv
│
└── figures/
    ├── hospitalization_distribution.png
    ├── daily_hospitalization_trend.png
    ├── hospitalizations_by_state.png
    ├── pm25_vs_hospitalizations.png
    ├── aqi_vs_hospitalizations.png
    ├── correlation_heatmap.png
    ├── monthly_hospitalization_pattern.png
    ├── hospitalizations_by_weekday.png
    ├── actual_vs_predicted.png
    ├── forecast_error_distribution.png
    ├── feature_importance.png
    ├── pollution_feature_importance.png
    └── future_hospitalization_forecast.png
```

---

# Public Health Applications

Although this project is a demonstration using synthetic data, the analytical framework illustrates several potential applications of environmental health data science.

### Hospital Capacity Planning

Forecasting potential changes in respiratory hospitalization demand could help healthcare organizations plan:

* Bed capacity
* Staffing
* Equipment
* Emergency response resources

### Environmental Health Surveillance

Combining air-quality indicators with healthcare utilization data can provide a framework for monitoring environmental health risks.

### Public Health Preparedness

Forecasting models can potentially support early identification of periods when respiratory healthcare demand may increase.

### Resource Allocation

Predictive analytics can inform the prioritization and allocation of limited healthcare resources.

### Environmental Health Policy

Environmental-health modeling can provide analytical evidence for investigating relationships between pollution exposure and healthcare demand.

---

# Important Limitations

This project should be interpreted as a **machine-learning and public-health analytics demonstration**, not as a validated clinical or epidemiological forecasting system.

### Synthetic Dataset

The dataset is synthetic. Its relationships and distributions are designed for modeling purposes and do not represent actual patient populations, hospitals, or disease surveillance systems.

### No Causal Inference

The model identifies predictive relationships. Feature importance and statistical associations should not be interpreted as evidence of causation.

### Environmental Forecast Assumptions

The demonstration future forecast uses recent environmental observations as baseline inputs. A production forecasting system should incorporate independently forecasted or real-time environmental measurements.

### Model Generalizability

Performance on synthetic data cannot be assumed to generalize to real-world populations, geographic regions, or healthcare systems.

### Uncertainty

The current implementation focuses primarily on point predictions and does not provide comprehensive probabilistic prediction intervals.

### External Validation

Real-world implementation would require extensive temporal, geographic, and population-level validation using independently collected data.

---

# Disclaimer

> **This project is intended strictly for educational, research, portfolio, and demonstration purposes.**
>
> The dataset used in this project is synthetic and does not contain real patient records, personally identifiable information, confidential healthcare information, or verified hospital surveillance data.
>
> The forecasts, model outputs, visualizations, feature importance results, and analytical conclusions generated by this project should **not** be interpreted as medical advice, clinical guidance, epidemiological evidence, or a validated healthcare decision-support system.
>
> The project does not establish causal relationships between air pollution and respiratory disease hospitalizations.
>
> No healthcare organization, clinician, public-health agency, or policymaker should rely on these model outputs for patient care, resource allocation, emergency response, public-health intervention, or policy decisions without appropriate validation using reliable real-world data.
>
> Anyone adapting this framework for real-world applications is responsible for independently validating the data, assumptions, methodology, model performance, uncertainty, fairness, privacy, security, and applicable regulatory requirements.

---

# Future Improvements

Future development could extend the project by incorporating:

* Real-world air-quality datasets
* Real-time environmental APIs
* Satellite-derived pollution indicators
* Meteorological forecasts
* Geographic Information System (GIS) analysis
* Hospital occupancy indicators
* Population mobility data
* Disease surveillance indicators
* XGBoost and LightGBM benchmarking
* ARIMA/SARIMA time-series models
* LSTM and transformer forecasting architectures
* SHAP-based explainable AI
* Prediction intervals
* Probabilistic forecasting
* Automated model monitoring
* Streamlit interactive dashboards
* Real-time public-health early-warning systems
* Multi-city forecasting
* Climate-change scenario analysis

---

# Reproducibility

The project is designed around a reproducible workflow:

```text
Dataset
   ↓
Data Validation
   ↓
Exploratory Analysis
   ↓
Feature Engineering
   ↓
Chronological Train/Test Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Feature Importance
   ↓
Future Forecasting
   ↓
Output Generation
```

A fixed `random_state=42` is used for the Random Forest model to improve reproducibility.

---

# Research & Analytical Value

This project demonstrates the intersection of:

**Environmental Data**

→ Air pollution and weather indicators

**Health Data**

→ Respiratory hospitalization demand

**Machine Learning**

→ Random Forest regression

**Time-Series Analytics**

→ Lag and rolling features

**Public Health Intelligence**

→ Forecasting and preparedness

The framework demonstrates how data science can move environmental-health analysis from retrospective reporting toward **predictive healthcare demand intelligence**.

---

# Author

**Jonah Buka**

Data Scientist | Machine Learning Engineer | Public Health Analytics

---

# License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for the complete license terms.

---

## Project Status

**Status:** Completed analytical prototype

The current version provides a complete exploratory and predictive modeling workflow using synthetic environmental-health data. Future versions can extend the framework toward real-world data integration, probabilistic forecasting, geospatial analytics, explainable AI, and interactive public-health decision-support applications.
