import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "hourly_energy.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# Forecast setup
LOOKBACK = 24       # Previous 24 hours
HORIZON = 1         # Predict the next hour
TARGET = "Global_active_power"

# Load hourly data
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"], index_col="timestamp")
df = df.sort_index()

# Identify continuous hourly segments
segment_id = df.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()

X_list = []
y_list = []
time_list = []

# Create sequences within each segment only
for _, segment in df.groupby(segment_id):
    values = segment.to_numpy(dtype=np.float32)
    target_col = segment.columns.get_loc(TARGET)

    # Require enough hours for input plus prediction target
    if len(segment) < LOOKBACK + HORIZON:
        continue

    for i in range(LOOKBACK, len(segment) - HORIZON + 1):
        X_list.append(values[i - LOOKBACK:i])
        y_list.append(values[i + HORIZON - 1, target_col])
        time_list.append(segment.index[i + HORIZON - 1])

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.float32)
target_times = pd.to_datetime(time_list)

print("Input sequences:", X.shape)
print("Target values:", y.shape)
print("First target timestamp:", target_times[0] if len(target_times) else "None")

# Chronological split: first 70% train, next 15% validation, last 15% test
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

# Fit scaler only on training input data to avoid leakage
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

def scale_sequences(data):
    shape = data.shape
    scaled = scaler.transform(data.reshape(-1, shape[-1]))
    return scaled.reshape(shape).astype(np.float32)

X_train = scale_sequences(X_train)
X_val = scale_sequences(X_val)
X_test = scale_sequences(X_test)

# Save arrays and scaler
np.savez_compressed(
    OUTPUT_DIR / "sequences.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    target_times=target_times.to_numpy(),
)

import joblib
joblib.dump(scaler, OUTPUT_DIR / "feature_scaler.pkl")

print("\nChronological split:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)
print("\nSaved sequences.npz and feature_scaler.pkl")