from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)
model = joblib.load('../model/xgb_fraud_model.pkl')
scaler = joblib.load('../model/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get("features")
        
        if not features or len(features) != 29:
            return jsonify({"error": "Input should be a list of 29 features."}), 400
        
        #normalize the input data to be 2D array
        X = np.array(features).reshape(1, -1)
        X[0, -1:] = scaler.transform([[X[0, -1]]])[0,0]
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1] 
        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

        