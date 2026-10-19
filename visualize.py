import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURE_DIR

def plot_predictions(results):
    plt.figure(figsize=(14, 6))
    plt.plot(results["date"], results["actual"], label="Actual")
    plt.plot(results["date"], results["predicted"], label="Predicted")
    plt.title("Actual vs Predicted Respiratory Hospitalizations")
    plt.xlabel("Date")
    plt.ylabel("Hospitalizations")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close()

def plot_residuals(results):
    plt.figure(figsize=(10, 6))
    sns.histplot(results["residual"], kde=True)
    plt.axvline(0, linestyle="--")
    plt.title("Forecast Residual Distribution")
    plt.xlabel("Actual - Predicted")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residual_distribution.png", dpi=300)
    plt.close()

def plot_future_forecast(forecast_df):
    aggregated = (
        forecast_df.groupby("date")["predicted_hospitalizations"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))
    plt.plot(
        aggregated["date"],
        aggregated["predicted_hospitalizations"],
        marker="o"
    )
    plt.title("Future Respiratory Hospitalization Forecast")
    plt.xlabel("Date")
    plt.ylabel("Predicted Hospitalizations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "future_forecast.png", dpi=300)
    plt.close()
