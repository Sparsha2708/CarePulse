"""
Maternal Health Risk Prediction - Flask API (FINAL)
─────────────────────────────────────────────────────
Start : py -3.10 predict_api.py
URL   : http://127.0.0.1:5000

Endpoints:
  GET  /         → health check
  GET  /test     → quick test prediction in browser
  POST /predict  → main prediction endpoint
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Load model files once at startup
# ─────────────────────────────────────────────
try:
    model         = joblib.load("risk_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: Run train_model.py first.")
    exit(1)

# Exact features the ML model was trained on
ML_FEATURES = ["Age", "SystolicBP", "DiastolicBP", "HeartRate", "BodyTemp"]

# ─────────────────────────────────────────────
# SpO2 threshold check (rule-based, not ML)
# MAX30102 sensor provides this value directly
# Clinical thresholds for pregnancy:
#   ≥ 95%  → Normal
#   90–94% → Warning (low oxygen)
#   < 90%  → Critical (dangerously low)
# ─────────────────────────────────────────────
def check_spo2(spo2):
    if spo2 is None:
        return None
    if spo2 >= 95:
        return "NORMAL"
    elif spo2 >= 90:
        return "WARNING"
    else:
        return "CRITICAL"

# ─────────────────────────────────────────────
# Combine ML result + SpO2 result
# SpO2 can only escalate risk, never reduce it
# ─────────────────────────────────────────────
def combine_alerts(ml_alert, spo2_alert):
    priority = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    if spo2_alert is None:
        return ml_alert
    # Return whichever is more severe
    if priority.get(spo2_alert, 0) > priority.get(ml_alert, 0):
        return spo2_alert
    return ml_alert

# ─────────────────────────────────────────────
# GET / — health check
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "CarePulse Maternal Health API is active",
        "endpoints": {
            "predict": "POST /predict",
            "test": "GET /test"
        }
    })

# ─────────────────────────────────────────────
# POST /predict — main prediction endpoint
#
# Required JSON fields:
#   Age, SystolicBP, DiastolicBP, HeartRate, BodyTemp
#
# Optional JSON field:
#   SpO2 (from MAX30102 — checked separately)
#
# Example:
# {
#   "Age": 28,
#   "SystolicBP": 130,
#   "DiastolicBP": 88,
#   "HeartRate": 95,
#   "BodyTemp": 37.2,
#   "SpO2": 96.5
# }
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Check all required ML fields are present
    missing = [f for f in ML_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # ── ML Prediction ──
        input_df      = pd.DataFrame([{f: data[f] for f in ML_FEATURES}])
        prediction    = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[0]
        risk_label    = label_encoder.inverse_transform(prediction)[0]

        alert_map = {
            "low risk":  {"alert": "NORMAL",   "color": "green",  "action": "No immediate action needed"},
            "mid risk":  {"alert": "WARNING",   "color": "orange", "action": "Monitor closely, consult doctor"},
            "high risk": {"alert": "CRITICAL",  "color": "red",    "action": "Seek immediate medical attention"}
        }
        ml_result = alert_map.get(risk_label.lower(), {
            "alert": "UNKNOWN", "color": "gray", "action": "Unable to classify"
        })

        # ── SpO2 Check ──
        spo2       = data.get("SpO2", None)
        spo2_alert = check_spo2(spo2)

        # ── Combine both ──
        final_alert = combine_alerts(ml_result["alert"], spo2_alert)

        # Update color and action based on final alert
        final_color = {"NORMAL": "green", "WARNING": "orange", "CRITICAL": "red"}.get(final_alert, "gray")
        final_action = {
            "NORMAL":   "No immediate action needed",
            "WARNING":  "Monitor closely, consult doctor soon",
            "CRITICAL": "Seek immediate medical attention"
        }.get(final_alert, "Unable to classify")

        return jsonify({
            "risk_level"   : risk_label,
            "ml_alert"     : ml_result["alert"],
            "spo2_alert"   : spo2_alert if spo2 is not None else "not provided",
            "final_alert"  : final_alert,
            "color"        : final_color,
            "action"       : final_action,
            "confidence"   : round(float(max(probabilities)) * 100, 2),
            "spo2_value"   : spo2,
            "probabilities": {
                label_encoder.classes_[i]: round(float(p) * 100, 2)
                for i, p in enumerate(probabilities)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# GET /test — quick browser test
# ─────────────────────────────────────────────
@app.route("/test", methods=["GET"])
def test():
    sample = {
        "Age": 35,
        "SystolicBP": 140,
        "DiastolicBP": 95,
        "HeartRate": 105,
        "BodyTemp": 37.8,
        "SpO2": 93.0
    }
    input_df   = pd.DataFrame([{f: sample[f] for f in ML_FEATURES}])
    prediction = model.predict(input_df)
    risk_label = label_encoder.inverse_transform(prediction)[0]
    spo2_alert = check_spo2(sample["SpO2"])
    alert_map  = {"low risk": "NORMAL", "mid risk": "WARNING", "high risk": "CRITICAL"}
    ml_alert   = alert_map.get(risk_label.lower(), "UNKNOWN")
    final      = combine_alerts(ml_alert, spo2_alert)

    return jsonify({
        "test_input" : sample,
        "ml_result"  : risk_label,
        "spo2_alert" : spo2_alert,
        "final_alert": final
    })

if __name__ == "__main__":
    print("Starting CarePulse API on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
