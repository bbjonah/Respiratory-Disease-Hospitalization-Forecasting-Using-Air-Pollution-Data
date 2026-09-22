import pandas as pd
from .config import DATA_PATH, TARGET

REQUIRED_COLUMNS = [
    "date", "state", "pm25", "pm10", "no2", "so2", "co", "ozone",
    "aqi", "temperature_c", "humidity_pct", "rainfall_mm",
    "wind_speed_kmh", "population_density", "smoking_prevalence_pct",
    "elderly_population_pct", TARGET
]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "state", TARGET])
    df["state"] = df["state"].astype(str)

    if df.duplicated(["date", "state"]).any():
        numeric = df.select_dtypes("number").columns
        agg = {c: "mean" for c in numeric}
        df = df.groupby(["date", "state"], as_index=False).agg(agg)

    return df.sort_values(["date", "state"]).reset_index(drop=True)
