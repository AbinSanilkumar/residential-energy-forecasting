import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_DIR / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "hourly_energy.csv"

# Create output folder if it does not exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load the semicolon-separated dataset; '?' represents missing values
df = pd.read_csv(
    RAW_FILE,
    sep=";",
    na_values=["?"],
    low_memory=False
)

# Combine Date and Time into one timestamp
df["timestamp"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

# Convert measurement columns to numeric
measurement_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

for col in measurement_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with invalid timestamps or missing target readings
df = df.dropna(subset=["timestamp", "Global_active_power"])

# Use timestamp as the index and aggregate minute readings into hourly means
df = df.set_index("timestamp")
hourly_df = df[measurement_cols].resample("h").mean()

# Remove hours with no valid readings
hourly_df = hourly_df.dropna(subset=["Global_active_power"])

# Save processed hourly dataset
hourly_df.to_csv(OUTPUT_FILE)

print("Preprocessing complete!")
print("Hourly dataset shape:", hourly_df.shape)
print("Saved to:", OUTPUT_FILE)
print("\nFirst five hourly rows:")
print(hourly_df.head())
print("\nMissing values per column:")
print(hourly_df.isna().sum())