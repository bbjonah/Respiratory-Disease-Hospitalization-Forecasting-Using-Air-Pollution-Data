import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

def evaluate_by_state(results):
    rows = []
    for state, group in results.groupby("state"):
        metrics = regression_metrics(
            group["actual"], group["predicted"]
        )
        rows.append({"state": state, **metrics})
    return pd.DataFrame(rows)
