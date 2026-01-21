from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import joblib
import os

app = Flask(__name__)
os.environ["MLFLOW_TRACKING_URI"] = ""

# -----------------------------
# Load MLflow Models (LOCAL FILE STORE)
# -----------------------------

flight_model = mlflow.pyfunc.load_model(
    "mlruns/1/models/m-aa460e5b997648d7a66c0c17b3836554/artifacts"
)

gender_model = mlflow.pyfunc.load_model(
    "mlruns/2/models/m-b2a42b7eb0e741f6be7960649941dab5/artifacts"
)

# -----------------------------
# Load Feature Lists & Encoders
# -----------------------------

expected_features = joblib.load("expected_features.pkl")
gender_expected_features = joblib.load("gender_expected_features.pkl")

# -----------------------------
# Home Route
# -----------------------------

@app.route("/")
def home():
    return {"message": "Voyage Analytics ML API is running"}

# -----------------------------
# Flight Price Prediction API
# -----------------------------

@app.route("/predict/flight", methods=["POST"])
def predict_flight_price():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # One-hot encode input
        df = pd.get_dummies(df)

        # Add missing columns
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # Align column order
        df = df[expected_features]

        prediction = flight_model.predict(df)

        return jsonify({
            "predicted_price": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Gender Classification API (Binary)
# -----------------------------

@app.route("/predict/gender", methods=["POST"])
def predict_gender():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # One-hot encode ONLY company
        df = pd.get_dummies(df)

        # Add missing columns
        for col in gender_expected_features:
            if col not in df.columns:
                df[col] = 0

        # Align column order
        df = df[gender_expected_features]

        prediction = gender_model.predict(df)

        gender_map = {0: "female", 1: "male"}
        predicted_gender = gender_map.get(int(prediction[0]), "unknown")

        return jsonify({"predicted_gender": predicted_gender})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Flask App
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
