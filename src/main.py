from .config import FIGURE_DIR, MODEL_DIR, OUTPUT_DIR
from .train import run_training
from .predict import recursive_forecast
from .visualize import (
    plot_predictions,
    plot_residuals,
    plot_future_forecast,
)

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, train, test, results = run_training()

    plot_predictions(results)
    plot_residuals(results)

    future = recursive_forecast()
    plot_future_forecast(future)

    print("\nPipeline completed successfully.")
    print(f"Model: {MODEL_DIR / 'respiratory_hospitalization_model.joblib'}")
    print(f"Predictions: {OUTPUT_DIR / 'forecast_predictions.csv'}")
    print(f"Future forecast: {OUTPUT_DIR / 'future_forecast.csv'}")

if __name__ == "__main__":
    main()
