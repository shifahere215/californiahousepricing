from flask import Flask, request, jsonify, render_template # type: ignore
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

app = Flask(__name__)

# Load pickled models and transformers
try:
    rf_model = pickle.load(open('rf_model.pkl', 'rb'))
    ridge_model = pickle.load(open('ridge_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    kmeans = pickle.load(open('kmeans.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading pickle files: {e}")
    exit(1)

# Load feature names from California housing dataset
california = fetch_california_housing()
feature_names = california.feature_names

# Hybrid prediction function
def predict_hybrid_new_data(new_data, rf_model, ridge_model, scaler, kmeans):
    """
    Transform and predict new data using the hybrid RandomForest + Ridge approach.
    
    Parameters:
    new_data: numpy array or pandas DataFrame of shape (n_samples, n_features)
    rf_model: trained RandomForestRegressor
    ridge_model: trained Ridge model for residuals
    scaler: trained RobustScaler
    kmeans: trained KMeans model for geographic clustering
    """
    # Convert new_data to DataFrame if it's a numpy array
    if isinstance(new_data, np.ndarray):
        new_data = pd.DataFrame(new_data, columns=feature_names)
    
    # Create interaction and polynomial terms
    new_data['MedInc_sq'] = new_data['MedInc'] ** 2
    new_data['MedInc_AveBedrms'] = new_data['MedInc'] * new_data['AveBedrms']
    new_data['MedInc_HouseAge'] = new_data['MedInc'] * new_data['HouseAge']
    
    # Apply KMeans clustering to geographic features
    geo_data = new_data[['Latitude', 'Longitude']]
    new_data['geo_cluster'] = kmeans.predict(geo_data)
    
    # Select top features for the model
    top_features = ['MedInc', 'MedInc_sq', 'Longitude', 'Latitude', 
                    'HouseAge', 'AveRooms', 'AveBedrms', 'MedInc_AveBedrms', 
                    'MedInc_HouseAge', 'geo_cluster']
    
    # Ensure all top_features exist in new_data
    missing_features = [f for f in top_features if f not in new_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in new_data: {missing_features}")
    
    X = new_data[top_features]
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Predict with RandomForest
    rf_predictions = rf_model.predict(X_scaled)
    
    # Predict residuals with Ridge
    ridge_predictions = ridge_model.predict(X_scaled)
    
    # Combine predictions
    final_predictions = rf_predictions + ridge_predictions
    
    return final_predictions

MODEL_ACCURACY = 0.7942

@app.route('/')
def home():
    return render_template('home.html', prediction_text='', model_accuracy=MODEL_ACCURACY)



@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get JSON data
        data = request.json['data']
        # Convert to DataFrame with feature names
        new_data = pd.DataFrame([data], columns=feature_names)
        # Make prediction
        output = predict_hybrid_new_data(new_data, rf_model, ridge_model, scaler, kmeans)
        return jsonify(float(output[0]))
    except Exception as e:
        return jsonify({'error': str(e)}), 400
        

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to float
        data = [float(x) for x in request.form.values()]
        # Convert to DataFrame with feature names
        new_data = pd.DataFrame([data], columns=feature_names)
        # Make prediction
        output = predict_hybrid_new_data(new_data, rf_model, ridge_model, scaler, kmeans)
    #     prediction = f"The House price prediction is {output[0]:.4f}"
    #     return render_template("home.html", prediction_text=prediction, model_accuracy=MODEL_ACCURACY)
    # except Exception as e:
    #     return render_template("home.html", prediction_text=f"Error: {str(e)}", model_accuracy=MODEL_ACCURACY)
        return render_template("home.html", prediction_text=f"The House price prediction is {output[0]:.4f}")
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)