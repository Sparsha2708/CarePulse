import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: Load dataset
# ─────────────────────────────────────────────
df = pd.read_csv("Maternal Health Risk Data Set.csv")
print(f"Dataset loaded: {len(df)} rows")

# Clean column names to strip out hidden spaces
df.columns = df.columns.str.strip()

# ─────────────────────────────────────────────
# STEP 2: Preprocess Features
# ─────────────────────────────────────────────
FEATURE_COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "HeartRate", "BodyTemp"]
TARGET = "RiskLevel"

# Convert BodyTemp Fahrenheit → Celsius safely
if df["BodyTemp"].mean() > 50:
    df["BodyTemp"] = ((df["BodyTemp"] - 32) * 5 / 9).round(1)
    print("✓ BodyTemp converted: °F → °C")

X = df[FEATURE_COLUMNS]
y = df[TARGET]

# ─────────────────────────────────────────────
# STEP 3: Encode labels and Scale Vitals
# ─────────────────────────────────────────────
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# We use a StandardScaler to ensure variance in BP doesn't swamp out small variances in Temp
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# STEP 4: Train / Test split (80/20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ─────────────────────────────────────────────
# STEP 5: Train Adjusted Random Forest
# Lowering depth and min_samples_leaf prevents it from forming extreme rule cutoffs
# ─────────────────────────────────────────────
print("\nTraining optimized model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,                # Lowered depth prevents hard cutoffs on normal numbers
    min_samples_split=10,
    min_samples_leaf=4,         # Forces trees to generalize broader groups
    class_weight="balanced",    # Prevents bias toward over-predicting high-risk groups
    random_state=42
)
model.fit(X_train, y_train)
print("Training complete.")

# ─────────────────────────────────────────────
# STEP 6: Evaluate Model
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  OPTIMIZED ACCURACY: {accuracy * 100:.2f}%")
print(f"{'='*40}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ─────────────────────────────────────────────
# STEP 7: Save Model, Encoder, AND Scaler
# ─────────────────────────────────────────────
joblib.dump(model, "risk_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl") # <-- CRITICAL: Must be used by predict_api.py!
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")

print("\n✓ Saved pipeline artifacts: risk_model.pkl, label_encoder.pkl, scaler.pkl")

# ─────────────────────────────────────────────
# STEP 8: Verification Checks
# ─────────────────────────────────────────────
print("\n── Live Calibration Test ──")
test_samples = [
    {"Age": 26, "SystolicBP": 120, "DiastolicBP": 80,  "HeartRate": 72,  "BodyTemp": 36.6}, # Your normal sample
    {"Age": 38, "SystolicBP": 145, "DiastolicBP": 95,  "HeartRate": 98,  "BodyTemp": 37.5}, # Clear hypertension risk
]

for idx, sample in enumerate(test_samples):
    sample_df = pd.DataFrame([sample])[FEATURE_COLUMNS]
    scaled_sample = scaler.transform(sample_df)
    pred_code = model.predict(scaled_sample)[0]
    result_text = label_encoder.inverse_transform([pred_code])[0]
    print(f"  Test Case {idx+1} (Age {sample['Age']}, BP {sample['SystolicBP']}/{sample['DiastolicBP']}) → Predicted: {result_text.upper()}")