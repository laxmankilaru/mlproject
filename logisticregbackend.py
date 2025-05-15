!pip install flask flask-cors pyngrok joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import joblib
import pandas as pd

# Load the model
pipe = joblib.load("model.pkl")

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "API is working"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # Respond to CORS preflight
        return jsonify({"message": "CORS preflight successful"}), 200

    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        pred = pipe.predict_proba(input_df)[0]
        return jsonify({
            "lose": round(pred[0] * 100, 1),
            "win": round(pred[1] * 100, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# **Set your ngrok authtoken**
ngrok.set_auth_token("2x5JmpIyuZZ23AcMkf3YG8qp1N6_4JqhewHUeP73GwebBdfow")  # Replace YOUR_AUTHTOKEN with your actual token

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print("✅ Public URL:", public_url)

# Run Flask app
app.run(port=5000)
