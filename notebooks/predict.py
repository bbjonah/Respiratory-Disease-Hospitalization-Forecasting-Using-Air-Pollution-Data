import numpy as np
import pandas as pd

from .config import TARGET, FORECAST_HORIZON, OUTPUT_DIR, MODEL_DIR
from .data import load_data
from .features import build_features, FEATURES
from .model import load_model

def recursive_forecast(horizon=FORECAST_HORIZON):
    model = load_model(
        MODEL_DIR / "respiratory_hospitalization_model.joblib"
    )

    history = load_data().sort_values(["state", "date"]).copy()

    forecasts = []

    for _ in range(horizon):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        new_rows = []

        for state in history["state"].unique():
            state_history = (
                history[history["state"] == state]
                .sort_values("date")
                .copy()
            )

            latest = state_history.iloc[-1]

            row = latest.copy()
            row["date"] = next_date

            # Environmental variables are held at the most recent
            # observed value in this demonstration scenario.
            row[TARGET] = np.nan

            new_rows.append(row)

        future_day = pd.DataFrame(new_rows)

        combined = pd.concat(
            [history, future_day],
            ignore_index=True
        )

        featured = build_features(combined)

        prediction_rows = featured[
            featured["date"] == next_date
        ].copy()

        valid_rows = prediction_rows.dropna(subset=FEATURES)

        predictions = model.predict(
            valid_rows[FEATURES]
        )

        for (_, row), prediction in zip(
            valid_rows.iterrows(),
            predictions
        ):
            forecasts.append({
                "date": next_date,
                "state": row["state"],
                "predicted_hospitalizations": max(
                    0.0,
                    float(prediction)
                ),
            })

            # Feed prediction into the next recursive step.
            history = pd.concat([
                history,
                pd.DataFrame([{
                    **row.to_dict(),
                    TARGET: max(0.0, float(prediction))
                }])
            ], ignore_index=True)

    forecast_df = pd.DataFrame(forecasts)

    forecast_df.to_csv(
        OUTPUT_DIR / "future_forecast.csv",
        index=False
    )

    return forecast_df
