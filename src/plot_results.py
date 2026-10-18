import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
PLOT_DIR = PROJECT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions and target timestamps
pred_data = np.load(DATA_DIR / "test_predictions.npz")
sequence_data = np.load(DATA_DIR / "sequences.npz")

actual = pred_data["actual"]
transformer_pred = pred_data["transformer_predictions"]
baseline_pred = pred_data["persistence_predictions"]

# Test timestamps are the final 15% of all sequence target timestamps
all_times = pd.to_datetime(sequence_data["target_times"])
test_start = int(len(all_times) * 0.85)
test_times = all_times[test_start:]

# Guard against any unexpected length mismatch
n = min(len(test_times), len(actual))
test_times = test_times[:n]
actual = actual[:n]
transformer_pred = transformer_pred[:n]
baseline_pred = baseline_pred[:n]

# --------------------------------------------------
# Plot 1: First week of test-set predictions
# --------------------------------------------------
hours_to_plot = min(168, n)

plt.figure(figsize=(14, 5))
plt.plot(
    test_times[:hours_to_plot],
    actual[:hours_to_plot],
    label="Actual consumption",
    linewidth=2
)
plt.plot(
    test_times[:hours_to_plot],
    transformer_pred[:hours_to_plot],
    label="Transformer prediction",
    linewidth=1.5
)
plt.plot(
    test_times[:hours_to_plot],
    baseline_pred[:hours_to_plot],
    label="Last-hour baseline",
    linewidth=1.2,
    alpha=0.8
)

plt.title("Hourly Household Power: Actual vs Predicted (Test Week)")
plt.xlabel("Time")
plt.ylabel("Global active power (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "test_week_predictions.png", dpi=300)
plt.show()

# --------------------------------------------------
# Plot 2: Actual vs Transformer predictions
# --------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(actual, transformer_pred, alpha=0.25, s=12)

lower = min(actual.min(), transformer_pred.min())
upper = max(actual.max(), transformer_pred.max())

plt.plot([lower, upper], [lower, upper], "r--", label="Perfect prediction")
plt.title("Actual vs Transformer Prediction")
plt.xlabel("Actual power (kW)")
plt.ylabel("Predicted power (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "actual_vs_predicted.png", dpi=300)
plt.show()

# --------------------------------------------------
# Print metrics again for convenience
# --------------------------------------------------
for name, pred in [
    ("Transformer", transformer_pred),
    ("Last-hour baseline", baseline_pred)
]:
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    print(f"{name}: MAE={mae:.4f} kW, RMSE={rmse:.4f} kW")

print("\nPlots saved in:", PLOT_DIR)