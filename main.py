# California Housing Price Prediction - Linear vs Non-Linear Models
# This script runs the complete analysis pipeline

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the California Housing dataset"""
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
    return data

def split_data(data):
    """Split data into train/validation/test sets"""
    # Define features and target
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']
    
    # Split the data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model"""
    print("Training Linear Regression model...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    lr_pipeline.fit(X_train, y_train)
    print("Linear Regression model trained successfully!")
    return lr_pipeline

def train_lasso_regression(X_train, y_train):
    """Train Lasso Regression model with hyperparameter tuning"""
    print("Training Lasso Regression model...")
    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso(random_state=42))
    ])
    
    # Perform hyperparameter tuning for alpha
    lasso_param_grid = {
        'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    lasso_grid_search = GridSearchCV(
        lasso_pipeline,
        lasso_param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    lasso_grid_search.fit(X_train, y_train)
    best_lasso_model = lasso_grid_search.best_estimator_
    
    print(f"Best Lasso parameters: {lasso_grid_search.best_params_}")
    print("Lasso Regression model trained successfully!")
    return best_lasso_model

def train_random_forest(X_train, y_train):
    """Train Random Forest model with hyperparameter tuning"""
    print("Training Random Forest model...")
    rf_pipeline = Pipeline([
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Perform hyperparameter tuning
    rf_param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__max_features': ['sqrt', 'auto']
    }
    
    rf_grid_search = GridSearchCV(
        rf_pipeline,
        rf_param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    
    print(f"Best Random Forest parameters: {rf_grid_search.best_params_}")
    print("Random Forest model trained successfully!")
    return best_rf_model

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """Evaluate model performance on all datasets"""
    # Make predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2
    
    train_rmse, train_mae, train_r2 = calculate_metrics(y_train, train_pred)
    val_rmse, val_mae, val_r2 = calculate_metrics(y_val, val_pred)
    test_rmse, test_mae, test_r2 = calculate_metrics(y_test, test_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  Train - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  Validation - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    print(f"  Test - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    return {
        'train': (train_rmse, train_mae, train_r2),
        'val': (val_rmse, val_mae, val_r2),
        'test': (test_rmse, test_mae, test_r2)
    }

def export_models(lr_model, lasso_model, rf_model, feature_names):
    """Export trained models and feature names"""
    print("\nExporting models...")
    joblib.dump(rf_model, 'california_housing_rf_model.pkl')
    print("Random Forest model saved as 'california_housing_rf_model.pkl'")
    
    joblib.dump(lasso_model, 'california_housing_lasso_model.pkl')
    print("Lasso model saved as 'california_housing_lasso_model.pkl'")
    
    joblib.dump(lr_model, 'california_housing_lr_model.pkl')
    print("Linear Regression model saved as 'california_housing_lr_model.pkl'")
    
    joblib.dump(feature_names, 'feature_names.pkl')
    print("Feature names saved as 'feature_names.pkl'")

def main():
    """Main function to run the complete analysis"""
    print("=" * 60)
    print("California Housing Price Prediction")
    print("Comparing Linear Regression, Lasso Regression, and Random Forest")
    print("=" * 60)
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    
    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    lasso_model = train_lasso_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate models
    lr_metrics = evaluate_model(lr_model, X_train, y_train, X_val, y_val, X_test, y_test, "Linear Regression")
    lasso_metrics = evaluate_model(lasso_model, X_train, y_train, X_val, y_val, X_test, y_test, "Lasso Regression")
    rf_metrics = evaluate_model(rf_model, X_train, y_train, X_val, y_val, X_test, y_test, "Random Forest")
    
    # Identify best model based on validation RMSE
    val_rmse_lr = lr_metrics['val'][0]
    val_rmse_lasso = lasso_metrics['val'][0]
    val_rmse_rf = rf_metrics['val'][0]
    
    best_model_name = ""
    if val_rmse_lr <= val_rmse_lasso and val_rmse_lr <= val_rmse_rf:
        best_model_name = "Linear Regression"
        best_model = lr_model
    elif val_rmse_lasso <= val_rmse_rf:
        best_model_name = "Lasso Regression"
        best_model = lasso_model
    else:
        best_model_name = "Random Forest"
        best_model = rf_model
    
    print(f"\nBest model based on validation RMSE: {best_model_name}")
    
    # Export models
    feature_names = list(data.drop('median_house_value', axis=1).columns)
    export_models(lr_model, lasso_model, rf_model, feature_names)
    
    print("\nAnalysis complete!")
    print("Models and results have been saved to the current directory.")

if __name__ == '__main__':
    main()