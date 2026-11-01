
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Model file path
MODEL_PATH = "student_result_model.pkl"

# Load trained ML model
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


# Health check route for GitHub Actions
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    }), 200


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model file not found"
        }), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Please provide JSON input"
            }), 400

        required_features = [
            "attendance",
            "internal_marks",
            "assignment_marks",
            "previous_score"
        ]

        # Validate input fields
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    "error": f"Missing required field: {feature}"
                }), 400

        # Prepare input data
        input_data = np.array([[
            float(data["attendance"]),
            float(data["internal_marks"]),
            float(data["assignment_marks"]),
            float(data["previous_score"])
        ]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Convert prediction into PASS or FAIL
        if isinstance(prediction, str):
            result = prediction.upper()
        else:
            result = "PASS" if int(prediction) == 1 else "FAIL"

        return jsonify({
            "prediction": result,
            "status": "success"
        }), 200

    except (ValueError, TypeError):
        return jsonify({
            "error": "All input values must be valid numbers"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# Run Flask application
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
