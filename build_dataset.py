"""
Maternal Health Dataset Builder
────────────────────────────────
This script does two things:
  1. Loads the base Kaggle maternal health dataset
  2. Adds medically accurate SpO2 (oxygen saturation) values
     based on pregnancy risk level and existing vitals
  3. Saves a clean, sensor-matched CSV for your ML model

SENSORS THIS DATASET MATCHES:
  MAX30102  → HeartRate, SpO2
  DS18B20   → BodyTemp (°C)
  BP Sensor → SystolicBP, DiastolicBP
  (Age is entered manually at registration)

Run:  python build_dataset.py
Output: maternal_health_sensor_dataset.csv
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ──────────────────────────────────────────────────────
# STEP 1: Load base dataset
# Download from: https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data
# ──────────────────────────────────────────────────────
print("Loading base dataset...")
df = pd.read_csv("Maternal Health Risk Data Set.csv")
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

# ──────────────────────────────────────────────────────
# STEP 2: Convert temperature F → C if needed
# Original dataset uses Fahrenheit. DS18B20 reads Celsius.
# Normal pregnancy body temp: 36.1–37.5°C
# ──────────────────────────────────────────────────────
if df["BodyTemp"].mean() > 50:  # it's in Fahrenheit
    df["BodyTemp"] = ((df["BodyTemp"] - 32) * 5 / 9).round(1)
    print("Converted BodyTemp: Fahrenheit → Celsius")

print(f"BodyTemp range: {df['BodyTemp'].min()}°C – {df['BodyTemp'].max()}°C")

# ──────────────────────────────────────────────────────
# STEP 3: Add SpO2 column
#
# Medical reference for SpO2 in pregnancy:
#   Normal (low risk):    96–100%  (mean ~98%)
#   Moderate (mid risk):  93–97%   (mean ~95%)
#   Critical (high risk): 88–94%   (mean ~91%)
#
# Also correlated with: high BP → lower SpO2
#                       high heart rate → slightly lower SpO2
# ──────────────────────────────────────────────────────
print("\nGenerating SpO2 values based on risk level and vitals...")

def generate_spo2(row):
    risk = row["RiskLevel"].lower()

    # Base SpO2 by risk category
    if "low" in risk:
        base = np.random.normal(loc=98.0, scale=1.0)
        base = np.clip(base, 96, 100)
    elif "mid" in risk:
        base = np.random.normal(loc=95.5, scale=1.2)
        base = np.clip(base, 93, 98)
    else:  # high risk
        base = np.random.normal(loc=91.5, scale=1.8)
        base = np.clip(base, 87, 95)

    # Reduce SpO2 slightly if BP is very high (preeclampsia effect)
    if row["SystolicBP"] > 140:
        base -= np.random.uniform(0.5, 2.0)

    # Reduce SpO2 slightly if heart rate is very elevated
    if row["HeartRate"] > 100:
        base -= np.random.uniform(0.3, 1.0)

    # Round to 1 decimal, stay in valid range
    return round(np.clip(base, 85, 100), 1)

df["SpO2"] = df.apply(generate_spo2, axis=1)
print("SpO2 added.")
print(df.groupby("RiskLevel")["SpO2"].describe().round(2))

# ──────────────────────────────────────────────────────
# STEP 4: Reorder columns to match sensor reading order
# (The order a nurse/health worker would see on screen)
# ──────────────────────────────────────────────────────
df = df[[
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "HeartRate",
    "SpO2",
    "BodyTemp",
    "BS",           # Blood Sugar
    "RiskLevel"     # Target label
]]

# ──────────────────────────────────────────────────────
# STEP 5: Clean & validate
# ──────────────────────────────────────────────────────
print("\nChecking for nulls...")
print(df.isnull().sum())

# Remove any rows with physiologically impossible values
before = len(df)
df = df[
    (df["SystolicBP"] >= 70)  & (df["SystolicBP"] <= 200)  &
    (df["DiastolicBP"] >= 40) & (df["DiastolicBP"] <= 130)  &
    (df["HeartRate"] >= 40)   & (df["HeartRate"] <= 160)    &
    (df["SpO2"] >= 80)        & (df["SpO2"] <= 100)         &
    (df["BodyTemp"] >= 35.0)  & (df["BodyTemp"] <= 42.0)    &
    (df["Age"] >= 10)         & (df["Age"] <= 70)
]
print(f"Removed {before - len(df)} invalid rows. Final: {len(df)} rows.")

# ──────────────────────────────────────────────────────
# STEP 6: Show summary stats
# ──────────────────────────────────────────────────────
print("\n── Final Dataset Summary ──")
print(f"Total samples : {len(df)}")
print(f"\nRisk distribution:")
print(df["RiskLevel"].value_counts())

print(f"\nFeature ranges:")
for col in df.columns[:-1]:
    print(f"  {col:15s}: {df[col].min():.1f} – {df[col].max():.1f}  (mean: {df[col].mean():.1f})")

# ──────────────────────────────────────────────────────
# STEP 7: Save
# ──────────────────────────────────────────────────────
output_file = "maternal_health_sensor_dataset.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")
print("Ready for use in train_model.py")

# ──────────────────────────────────────────────────────
# STEP 8: Preview first 5 rows
# ──────────────────────────────────────────────────────
print("\nSample rows:")
print(df.head())
