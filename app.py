from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained XGBoost model using pickle
with open("xgboost_landslide_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.get_json()

    # Extract features from input data
    features = [
        input_data['elevation'],
        input_data['slope'],
        input_data['rainfall'],
        input_data['temperature'],
        input_data['soil_moisture'],
        input_data['tpi'],
        input_data['twi'],
        input_data['earthquake'],
        input_data['distance_to_river'],
        input_data['vegetation']
    ]
    
    # Convert features into a numpy array (as required by the model)
    features = np.array(features).reshape(1, -1)

    # Make prediction and calculate probability
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # Return prediction and probability as JSON response
    return jsonify({'prediction': int(prediction[0]), 'probability': round(probability, 2)})

if __name__ == '__main__':
    app.run(debug=True)
