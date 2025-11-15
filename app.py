from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
import os

app = Flask(__name__)

# Global variables to store models and feature names
models = {}
feature_names = []
model_metrics = {}

# Load models and data on startup
def load_models():
    global models, feature_names, model_metrics
    
    try:
        # Load trained models
        models['linear'] = joblib.load('california_housing_lr_model.pkl')
        models['lasso'] = joblib.load('california_housing_lasso_model.pkl')
        models['random_forest'] = joblib.load('california_housing_rf_model.pkl')
        
        # Load feature names
        feature_names = joblib.load('feature_names.pkl')
        
        # Model performance metrics (from your results)
        model_metrics = {
            'linear': {
                'train': {'rmse': 0.6695, 'mae': 0.4811, 'r2': 0.6641},
                'validation': {'rmse': 0.6636, 'mae': 0.4855, 'r2': 0.6708},
                'test': {'rmse': 0.6803, 'mae': 0.4916, 'r2': 0.6469}
            },
            'lasso': {
                'train': {'rmse': 0.6695, 'mae': 0.4812, 'r2': 0.6641},
                'validation': {'rmse': 0.6638, 'mae': 0.4855, 'r2': 0.6706},
                'test': {'rmse': 0.6803, 'mae': 0.4916, 'r2': 0.6468}
            },
            'random_forest': {
                'train': {'rmse': 0.2390, 'mae': 0.1595, 'r2': 0.9572},
                'validation': {'rmse': 0.5227, 'mae': 0.3531, 'r2': 0.7958},
                'test': {'rmse': 0.5184, 'mae': 0.3523, 'r2': 0.7950}
            }
        }
        
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Get feature statistics for input validation
def get_feature_stats():
    california_housing = fetch_california_housing()
    data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    
    # Add derived features
    data['rooms_per_household'] = data['AveRooms'] / data['AveOccup']
    data['bedrooms_ratio'] = data['AveBedrms'] / data['AveRooms']
    data['population_per_household'] = data['Population'] / data['AveOccup']
    data['rooms_per_person'] = data['AveRooms'] / data['Population']
    
    # Handle potential division by zero
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.median(), inplace=True)
    
    stats = {}
    for col in data.columns:
        stats[col] = {
            'min': float(data[col].min()),
            'max': float(data[col].max()),
            'mean': float(data[col].mean())
        }
    
    return stats

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_data = {}
        for feature in feature_names:
            input_data[feature] = float(request.form.get(feature, 0))
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make predictions with all models
        predictions = {}
        for model_name, model in models.items():
            pred = model.predict(input_df)[0]
            predictions[model_name] = round(pred, 2)
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/model_metrics')
def api_model_metrics():
    return jsonify(model_metrics)

@app.route('/api/feature_stats')
def api_feature_stats():
    return jsonify(get_feature_stats())

if __name__ == '__main__':
    if load_models():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please make sure model files exist.")