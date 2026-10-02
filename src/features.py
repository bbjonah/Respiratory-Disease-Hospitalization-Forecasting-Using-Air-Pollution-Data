import numpy as np
import pandas as pd
from .config import TARGET

def build_features(df):
    data = df.copy().sort_values(["state", "date"])

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["dayofweek"] = data["date"].dt.dayofweek
    data["dayofyear"] = data["date"].dt.dayofyear

    # Cyclical calendar encoding
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)
    data["doy_sin"] = np.sin(2 * np.pi * data["dayofyear"] / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * data["dayofyear"] / 365.25)

    grouped_target = data.groupby("state")[TARGET]

    for lag in [1, 7, 14, 30]:
        data[f"lag_{lag}"] = grouped_target.shift(lag)

    for window in [7, 14, 30]:
        data[f"rolling_{window}"] = (
            grouped_target.transform(
                lambda x: x.shift(1).rolling(window).mean()
            )
        )

    data["rolling_std_7"] = (
        grouped_target.transform(
            lambda x: x.shift(1).rolling(7).std()
        )
    )

    # Pollution history: only information available before prediction date.
    for col in ["pm25", "pm10", "no2", "so2", "co", "ozone", "aqi"]:
        data[f"{col}_lag_1"] = data.groupby("state")[col].shift(1)
        data[f"{col}_rolling_7"] = (
            data.groupby("state")[col]
            .transform(lambda x: x.shift(1).rolling(7).mean())
        )

    data["particulate_burden"] = data["pm25"] + data["pm10"]
    data["pm25_aqi_interaction"] = data["pm25"] * data["aqi"]

    return data


NUMERIC_FEATURES = [
    "pm25", "pm10", "no2", "so2", "co", "ozone", "aqi",
    "temperature_c", "humidity_pct", "rainfall_mm", "wind_speed_kmh",
    "population_density", "smoking_prevalence_pct",
    "elderly_population_pct", "year", "month", "quarter",
    "dayofweek", "dayofyear", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_7", "rolling_14", "rolling_30", "rolling_std_7",
    "particulate_burden", "pm25_aqi_interaction",
    "pm25_lag_1", "pm25_rolling_7",
    "pm10_lag_1", "pm10_rolling_7",
    "no2_lag_1", "no2_rolling_7",
    "so2_lag_1", "so2_rolling_7",
    "co_lag_1", "co_rolling_7",
    "ozone_lag_1", "ozone_rolling_7",
    "aqi_lag_1", "aqi_rolling_7",
]

CATEGORICAL_FEATURES = ["state"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
