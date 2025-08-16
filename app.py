from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("fraud_model.pkl")

# Feature columns expected by the model
FEATURE_COLUMNS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
    'V28', 'Amount'
]

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Credit Card Fraud Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON"}), 400

        features = data["features"]

        if len(features) != len(FEATURE_COLUMNS):
            return jsonify({
                "error": f"Expected {len(FEATURE_COLUMNS)} features, got {len(features)}"
            }), 400

        # Convert to DataFrame with column names
        df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        # Predict class and probability
        prediction = model.predict(df)[0]
        fraud_proba = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": float(fraud_proba)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
