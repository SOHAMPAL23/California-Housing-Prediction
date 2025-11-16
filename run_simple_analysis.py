# Simple California Housing Price Prediction Analysis
# This script runs a simplified version of the analysis

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading California Housing dataset...")
    california_housing = fetch_california_housing()
    
    # Convert to DataFrame
    data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    data['median_house_value'] = california_housing.target
    
    # Create new interaction features
    data['rooms_per_household'] = data['AveRooms'] / data['AveOccup']
    data['bedrooms_ratio'] = data['AveBedrms'] / data['AveRooms']
    data['population_per_household'] = data['Population'] / data['AveOccup']
    data['rooms_per_person'] = data['AveRooms'] / data['Population']
    
    # Handle potential division by zero
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.median(), inplace=True)
    
    print(f"Dataset loaded with shape: {data.shape}")
    
    # Define features and target
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']
    
    # Split the data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train Linear Regression
    print("\nTraining Linear Regression model...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    
    # Train Lasso Regression
    print("Training Lasso Regression model...")
    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=0.001, random_state=42))
    ])
    lasso_pipeline.fit(X_train, y_train)
    
    # Train Random Forest (with simplified parameters for faster execution)
    print("Training Random Forest model...")
    rf_pipeline = Pipeline([
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate models
    def evaluate_model(model, X, y, model_name):
        pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        return rmse, mae, r2
    
    # Test set evaluation
    lr_rmse, lr_mae, lr_r2 = evaluate_model(lr_pipeline, X_test, y_test, "Linear Regression")
    lasso_rmse, lasso_mae, lasso_r2 = evaluate_model(lasso_pipeline, X_test, y_test, "Lasso Regression")
    rf_rmse, rf_mae, rf_r2 = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
    
    print("\nModel Performance on Test Set:")
    print(f"Linear Regression  - RMSE: {lr_rmse:.4f}, MAE: {lr_mae:.4f}, R²: {lr_r2:.4f}")
    print(f"Lasso Regression   - RMSE: {lasso_rmse:.4f}, MAE: {lasso_mae:.4f}, R²: {lasso_r2:.4f}")
    print(f"Random Forest      - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")
    
    # Export models
    print("\nExporting models...")
    joblib.dump(rf_pipeline, 'california_housing_rf_model.pkl')
    joblib.dump(lasso_pipeline, 'california_housing_lasso_model.pkl')
    joblib.dump(lr_pipeline, 'california_housing_lr_model.pkl')
    feature_names = list(X.columns)
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("Models exported successfully!")
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()