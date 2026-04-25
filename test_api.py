"""
CarePulse API Test Script (FINAL)
──────────────────────────────────
Run WHILE predict_api.py is running in another terminal.

  Terminal 1: py -3.10 predict_api.py   ← keep open
  Terminal 2: py -3.10 test_api.py      ← run this
"""

import requests

API_URL = "http://127.0.0.1:5000/predict"

# These 5 are what the ML model uses
# SpO2 is optional — sent separately for threshold check
test_patients = [
    {
        "label": "Healthy young mother (expect: LOW RISK)",
        "data": {
            "Age": 22,
            "SystolicBP": 108,
            "DiastolicBP": 68,
            "HeartRate": 74,
            "BodyTemp": 36.5,
            "SpO2": 98.5        # normal oxygen
        }
    },
    {
        "label": "Moderate concern (expect: MID RISK)",
        "data": {
            "Age": 33,
            "SystolicBP": 132,
            "DiastolicBP": 87,
            "HeartRate": 94,
            "BodyTemp": 37.2,
            "SpO2": 94.0        # slightly low oxygen → escalates alert
        }
    },
    {
        "label": "High risk / preeclampsia signs (expect: CRITICAL)",
        "data": {
            "Age": 42,
            "SystolicBP": 158,
            "DiastolicBP": 104,
            "HeartRate": 112,
            "BodyTemp": 38.3,
            "SpO2": 91.0        # dangerously low oxygen
        }
    },
    {
        "label": "Low vitals but low SpO2 (SpO2 escalates alert)",
        "data": {
            "Age": 25,
            "SystolicBP": 112,
            "DiastolicBP": 72,
            "HeartRate": 80,
            "BodyTemp": 36.8,
            "SpO2": 88.0        # critical SpO2 should escalate to CRITICAL
        }
    }
]

print("=" * 50)
print("  CAREPULSE — API TEST")
print("=" * 50)

for patient in test_patients:
    print(f"\n{'-' * 50}")
    print(f"  {patient['label']}")
    print(f"  Input: {patient['data']}")

    try:
        response = requests.post(API_URL, json=patient["data"])
        result   = response.json()

        print(f"\n  ML Result  : {result.get('risk_level', 'N/A').upper()}")
        print(f"  ML Alert   : {result.get('ml_alert', 'N/A')}")
        print(f"  SpO2 Alert : {result.get('spo2_alert', 'N/A')} (SpO2: {result.get('spo2_value')}%)")
        print(f"  FINAL ALERT: {result.get('final_alert', 'N/A')} ← what dashboard shows")
        print(f"  Action     : {result.get('action', 'N/A')}")
        print(f"  Confidence : {result.get('confidence', 'N/A')}%")

    except requests.exceptions.ConnectionError:
        print("\n  ERROR: API not running.")
        print("  → Start Terminal 1: py -3.10 predict_api.py")
        break
