"""
Maternal Health Risk Prediction - Model Training (FINAL)
─────────────────────────────────────────────────────────
Dataset : Maternal Health Risk Data Set.csv (Kaggle)
Features: Age, SystolicBP, DiastolicBP, HeartRate, BodyTemp
Target  : RiskLevel (low risk / mid risk / high risk)

Run ONCE: py -3.10 train_model.py
Outputs : risk_model.pkl, label_encoder.pkl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: Load dataset
# ─────────────────────────────────────────────
df = pd.read_csv("Maternal Health Risk Data Set.csv")
print(f"Dataset loaded: {len(df)} rows")
print(f"Columns found : {list(df.columns)}")

print(f"\nRisk distribution:")
print(df["RiskLevel"].value_counts())

# ─────────────────────────────────────────────
# STEP 2: Features — exactly what your sensors measure
#   Age        → entered manually by health worker
#   SystolicBP → BP sensor
#   DiastolicBP→ BP sensor
#   HeartRate  → MAX30102 sensor
#   BodyTemp   → DS18B20 sensor (convert F to C below)
#
#   NOT included:
#   BS (Blood Sugar) → no glucometer in hardware
#   SpO2             → handled separately via threshold rules
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "HeartRate",
    "BodyTemp"
]
TARGET = "RiskLevel"

# Convert BodyTemp Fahrenheit → Celsius (DS18B20 reads Celsius)
if df["BodyTemp"].mean() > 50:
    df["BodyTemp"] = ((df["BodyTemp"] - 32) * 5 / 9).round(1)
    print("\nBodyTemp converted: °F → °C")

X = df[FEATURE_COLUMNS]
y = df[TARGET]

print(f"\nFeature ranges:")
for col in FEATURE_COLUMNS:
    print(f"  {col:15s}: {df[col].min():.1f} – {df[col].max():.1f}")

# ─────────────────────────────────────────────
# STEP 3: Encode labels
# high risk → 0, low risk → 1, mid risk → 2
# ─────────────────────────────────────────────
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nLabel encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i} → {label}")

# ─────────────────────────────────────────────
# STEP 4: Train / Test split (80/20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 5: Train Random Forest
# Better than Decision Tree: more accurate, less overfitting
# ─────────────────────────────────────────────
print("\nTraining model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)
print("Training complete.")

# ─────────────────────────────────────────────
# STEP 6: Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  ACCURACY: {accuracy * 100:.2f}%")
print(f"{'='*40}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importance
print("Sensor importance (which reading matters most):")
for feat, imp in sorted(zip(FEATURE_COLUMNS, model.feature_importances_),
                         key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {feat:15s} {bar} {imp:.3f}")

# ─────────────────────────────────────────────
# STEP 7: Save
# ─────────────────────────────────────────────
joblib.dump(model, "risk_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")

print("\n✓ Saved: risk_model.pkl")
print("✓ Saved: label_encoder.pkl")
print("✓ Saved: feature_columns.pkl")

# ─────────────────────────────────────────────
# STEP 8: Quick sample predictions
# ─────────────────────────────────────────────
print("\n── Sample Predictions ──")
samples = [
    {"label": "Healthy",  "Age": 22, "SystolicBP": 110, "DiastolicBP": 70,  "HeartRate": 76,  "BodyTemp": 36.5},
    {"label": "Moderate", "Age": 34, "SystolicBP": 132, "DiastolicBP": 88,  "HeartRate": 95,  "BodyTemp": 37.2},
    {"label": "Critical", "Age": 42, "SystolicBP": 158, "DiastolicBP": 105, "HeartRate": 112, "BodyTemp": 38.3},
]
for s in samples:
    label = s.pop("label")
    pred = model.predict(pd.DataFrame([s]))[0]
    result = label_encoder.inverse_transform([pred])[0]
    print(f"  {label:10s} → {result.upper()}")

print("\nML training complete. Run predict_api.py next.")
