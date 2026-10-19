import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from .config import TARGET, RANDOM_STATE, TEST_SIZE, OUTPUT_DIR, MODEL_DIR
from .data import load_data
from .features import build_features, FEATURES
from .model import make_model, save_model
from .evaluate import regression_metrics, evaluate_by_state

def chronological_split(df):
    dates = np.sort(df["date"].unique())
    cutoff = dates[int(len(dates) * (1 - TEST_SIZE))]
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test

def run_training():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    raw = load_data()
    data = build_features(raw).dropna(subset=FEATURES + [TARGET]).copy()

    train, test = chronological_split(data)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = make_model()

    param_grid = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [8, 12, 16, None],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.6, 1.0],
    }

    # CV operates on the chronological training block only.
    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=12,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    metrics = regression_metrics(y_test, predictions)

    results = test[["date", "state", TARGET]].copy()
    results = results.rename(columns={TARGET: "actual"})
    results["predicted"] = predictions
    results["residual"] = results["actual"] - results["predicted"]

    results.to_csv(
        OUTPUT_DIR / "forecast_predictions.csv",
        index=False
    )

    regional = evaluate_by_state(results)
    regional.to_csv(
        OUTPUT_DIR / "regional_model_performance.csv",
        index=False
    )

    with open(OUTPUT_DIR / "model_metrics.json", "w") as f:
        json.dump(
            {
                "test_metrics": metrics,
                "best_parameters": search.best_params_,
                "training_rows": len(train),
                "test_rows": len(test),
                "test_start": str(test["date"].min()),
                "test_end": str(test["date"].max()),
            },
            f,
            indent=2,
        )

    save_model(
        best_model,
        MODEL_DIR / "respiratory_hospitalization_model.joblib"
    )

    print("\nMODEL PERFORMANCE")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nBest parameters:")
    print(search.best_params_)

    return best_model, train, test, results
