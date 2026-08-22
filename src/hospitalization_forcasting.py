# ==========================================================
# Respiratory Disease Hospitalization Forecasting
# Using Air Pollution Data
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    "respiratory_disease_hospitalization_forecasting_dataset.csv"
)

print(df.head())
print(df.shape)

# ==========================================================
# DATE PROCESSING
# ==========================================================

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["quarter"] = df["date"].dt.quarter

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

# Lag Features

df["lag_1"] = df.groupby("state")[
    "respiratory_hospitalizations"
].shift(1)

df["lag_7"] = df.groupby("state")[
    "respiratory_hospitalizations"
].shift(7)

df["lag_30"] = df.groupby("state")[
    "respiratory_hospitalizations"
].shift(30)

# Rolling Mean Features

df["rolling_7"] = (
    df.groupby("state")[
        "respiratory_hospitalizations"
    ]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

df["rolling_30"] = (
    df.groupby("state")[
        "respiratory_hospitalizations"
    ]
    .transform(
        lambda x: x.shift(1).rolling(30).mean()
    )
)

# Remove Nulls from Lag Creation

df.dropna(inplace=True)

# ==========================================================
# EXPLORATORY DATA ANALYSIS
# ==========================================================

plt.figure(figsize=(12,6))

monthly_trend = (
    df.groupby("date")[
        "respiratory_hospitalizations"
    ]
    .mean()
)

monthly_trend.plot()

plt.title("Average Daily Respiratory Hospitalizations")
plt.xlabel("Date")
plt.ylabel("Hospitalizations")
plt.show()

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(14,10))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# ENCODE STATE
# ==========================================================

encoder = LabelEncoder()

df["state_encoded"] = encoder.fit_transform(
    df["state"]
)

# ==========================================================
# FEATURE SELECTION
# ==========================================================

features = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "ozone",
    "aqi",
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "wind_speed_kmh",
    "population_density",
    "smoking_prevalence_pct",
    "elderly_population_pct",
    "year",
    "month",
    "dayofweek",
    "quarter",
    "state_encoded",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_7",
    "rolling_30"
]

target = "respiratory_hospitalizations"

X = df[features]
y = df[target]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================================
# MODEL
# ==========================================================

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# EVALUATION
# ==========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n========================")
print("MODEL PERFORMANCE")
print("========================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

plt.figure(figsize=(14,6))

plt.plot(
    results["Actual"].values[:300],
    label="Actual"
)

plt.plot(
    results["Predicted"].values[:300],
    label="Predicted"
)

plt.legend()

plt.title(
    "Actual vs Predicted Hospitalizations"
)

plt.show()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop Features")

print(importance.head(15))

plt.figure(figsize=(10,8))

sns.barplot(
    data=importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Feature Importances"
)

plt.show()

# ==========================================================
# SAVE RESULTS
# ==========================================================

results.to_csv(
    "forecast_predictions.csv",
    index=False
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "- forecast_predictions.csv"
)

print(
    "- feature_importance.csv"
)

# ==========================================================
# FUTURE FORECAST EXAMPLE
# ==========================================================

latest_data = X.tail(30)

future_forecasts = model.predict(
    latest_data
)

future_df = pd.DataFrame({
    "Forecasted_Hospitalizations":
        future_forecasts
})

future_df.to_csv(
    "future_forecast.csv",
    index=False
)

print(
    "\nFuture forecast file saved."
)

# ==========================================================
# END
# ==========================================================
