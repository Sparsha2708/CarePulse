"""
Maternal Health Risk Prediction - Flask API (UPDATED)
─────────────────────────────────────────────────────
Start : py -3.10 predict_api.py
URL   : http://127.0.0.1:5000

Endpoints:
  GET  /          → Health check
  GET  /latest    → Dashboard polls this for continuous live ESP32 readings
  POST /update    → ESP32 posts raw sensor stream values here
  POST /predict   → Main prediction endpoint (uses model + scaler)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import datetime

app = Flask(__name__)
CORS(app)  # Unlocks communication across different local server ports

# ─────────────────────────────────────────────
# Load pipeline components once at startup
# ─────────────────────────────────────────────
try:
    model         = joblib.load("risk_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler        = joblib.load("scaler.pkl")  # Critical for input calibration
    print("✓ Model, Label Encoder, and Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Missing pipeline files. Please run train_model.py first. ({e})")
    exit(1)

# Exact features and ordering the ML model expects
ML_FEATURES = ["Age", "SystolicBP", "DiastolicBP", "HeartRate", "BodyTemp"]

# Global store for the live ESP32 data stream
latest_reading = {}

# ─────────────────────────────────────────────
# SpO2 Threshold Check (Rule-based)
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
# Combine ML Result + SpO2 Risk Escalation
# ─────────────────────────────────────────────
def combine_alerts(ml_alert, spo2_alert):
    priority = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    if spo2_alert is None:
        return ml_alert
    if priority.get(spo2_alert, 0) > priority.get(ml_alert, 0):
        return spo2_alert
    return ml_alert

# ─────────────────────────────────────────────
# GET / — Health Check
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "CarePulse Maternal Health Monitor API is active",
        "endpoints": {
            "predict": "POST /predict",
            "update": "POST /update",
            "latest": "GET /latest"
        }
    })

# ─────────────────────────────────────────────
# POST /update — ESP32 posts data here
# ─────────────────────────────────────────────
@app.route("/update", methods=["POST"])
def update():
    global latest_reading
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty payload received"}), 400
            
        latest_reading = data
        latest_reading["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")
        return jsonify({"status": "saved", "timestamp": latest_reading["timestamp"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# GET /latest — Dashboard polls this for vitals
# ─────────────────────────────────────────────
@app.route("/latest", methods=["GET"])  
def latest():
    if not latest_reading:
        return jsonify({"status": "no_data"}), 200
    return jsonify(latest_reading)

# ─────────────────────────────────────────────
# POST /predict — Main Machine Learning Assessment
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Confirm all necessary ML features exist in request payload
    missing = [f for f in ML_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing operational metrics: {missing}"}), 400

    try:
        # 1. Structure raw data into an ordered DataFrame dataframe matching pipeline constraints
        input_df = pd.DataFrame([{f: data[f] for f in ML_FEATURES}])
        
        # 2. Apply your trained Scaler transform step 
        scaled_input = scaler.transform(input_df)
        
        # 3. Process machine learning evaluation tracks
        prediction    = model.predict(scaled_input)
        probabilities = model.predict_proba(scaled_input)[0]
        risk_label    = label_encoder.inverse_transform(prediction)[0]

        alert_map = {
            "low risk":  {"alert": "NORMAL",   "color": "green",  "action": "No immediate action needed"},
            "mid risk":  {"alert": "WARNING",  "color": "orange", "action": "Monitor closely, consult doctor soon"},
            "high risk": {"alert": "CRITICAL", "color": "red",    "action": "Seek immediate medical attention"}
        }
        ml_result = alert_map.get(risk_label.lower(), {
            "alert": "UNKNOWN", "color": "gray", "action": "Unable to safely classify profile parameters"
        })

        # 4. Process secondary rule-based SpO2 tracking step
        spo2       = data.get("SpO2", None)
        spo2_alert = check_spo2(spo2)

        # 5. Safety Merge: Escalate priority tier matrix based on oxygen limits if necessary
        final_alert = combine_alerts(ml_result["alert"], spo2_alert)

        final_color = {"NORMAL": "green", "WARNING": "orange", "CRITICAL": "red"}.get(final_alert, "gray")
        final_action = {
            "NORMAL":   "No immediate action needed. Continue routine checks.",
            "WARNING":  "Monitor closely, establish baseline patterns, consult doctor soon.",
            "CRITICAL": "Seek immediate medical attention or institutional care checkups."
        }.get(final_alert, "Unable to resolve clear diagnostic feedback action.")

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
        return jsonify({"error": f"Internal pipeline evaluation crash: {str(e)}"}), 500

# ─────────────────────────────────────────────
# GET /test — Quick test diagnostic utility endpoint
# ─────────────────────────────────────────────
@app.route("/test", methods=["GET"])
def test():
    sample = {
        "Age": 26,
        "SystolicBP": 120,
        "DiastolicBP": 80,
        "HeartRate": 72,
        "BodyTemp": 36.6,
        "SpO2": 98.0
    }
    input_df = pd.DataFrame([{f: sample[f] for f in ML_FEATURES}])
    scaled_sample = scaler.transform(input_df)
    
    prediction = model.predict(scaled_sample)
    risk_label = label_encoder.inverse_transform(prediction)[0]
    
    return jsonify({
        "message": "API system self-calibration diagnostic test complete.",
        "test_input_sample": sample,
        "calibrated_ml_output": risk_label.upper()
    })

if __name__ == "__main__":
    print("Starting CarePulse Machine Learning API Gateway Engine...")
    app.run(host="0.0.0.0", port=5000, debug=False)