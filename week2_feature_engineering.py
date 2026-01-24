#!/usr/bin/env python
"""
Week 2: Feature Engineering and Model Preparation for California Housing Prediction
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the California Housing dataset"""
    print("=" * 60)
    print("WEEK 2: FEATURE ENGINEERING AND MODEL PREPARATION")
    print("=" * 60)
    
    # Load the dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    df.rename(columns={'target': 'median_house_value'}, inplace=True)
    
    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df

def create_derived_features(df):
    """Create derived features from existing ones"""
    print(f"\n1. Creating Derived Features:")
    
    original_cols = df.columns.tolist()
    
    # Create new features based on domain knowledge
    df['rooms_per_household'] = df['AveRooms'] / df['HouseAge']  # Rooms per household
    df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']  # Bedrooms per room ratio
    df['population_per_household'] = df['Population'] / df['AveOccup']  # Population per household
    df['rooms_per_person'] = df['AveRooms'] / df['AveOccup']  # Rooms per person
    df['bedrooms_per_person'] = df['AveBedrms'] / df['AveOccup']  # Bedrooms per person
    df['household_size'] = df['Population'] / df['AveOccup']  # Household size
    
    # Location-based features
    df['lat_abs'] = df['Latitude'].abs()  # Absolute latitude
    df['long_abs'] = df['Longitude'].abs()  # Absolute longitude
    
    # Income-related ratios
    df['income_per_person'] = df['MedInc'] / df['AveOccup']  # Income per person
    
    # Area density indicators
    df['pop_density_proxy'] = df['Population'] / (df['AveOccup'] * df['HouseAge'])  # Population density proxy
    
    new_features = [col for col in df.columns if col not in original_cols]
    print(f"   ✅ Created {len(new_features)} new features:")
    for feature in new_features:
        print(f"      - {feature}")
    
    return df, new_features

def split_data(df):
    """Split data into train and test sets"""
    print(f"\n2. Splitting Data into Train/Test Sets:")
    
    # Define features and target
    feature_cols = [col for col in df.columns if col != 'median_house_value']
    X = df[feature_cols]
    y = df['median_house_value']
    
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   ✅ Data split successfully:")
    print(f"      - Training set: {X_train.shape[0]:,} samples")
    print(f"      - Test set: {X_test.shape[0]:,} samples")
    print(f"      - Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def apply_scaling(X_train, X_test):
    """Apply feature scaling for Linear Regression and Lasso"""
    print(f"\n3. Applying Feature Scaling:")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"   ✅ Feature scaling applied successfully")
    print(f"      - Training set shape after scaling: {X_train_scaled.shape}")
    print(f"      - Test set shape after scaling: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler

def train_linear_regression_baseline(X_train, y_train, X_test, y_test):
    """Train Linear Regression baseline model"""
    print(f"\n4. Training Linear Regression Baseline Model:")
    
    # Initialize and train the model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"   ✅ Linear Regression trained successfully")
    print(f"   📊 Training Metrics:")
    print(f"      - RMSE: {train_rmse:.4f}")
    print(f"      - R² Score: {train_r2:.4f}")
    print(f"   📊 Test Metrics:")
    print(f"      - RMSE: {test_rmse:.4f}")
    print(f"      - R² Score: {test_r2:.4f}")
    
    return lr_model, y_pred_test, test_rmse, test_r2

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance based on coefficients"""
    print(f"\n5. Feature Importance Analysis (Linear Regression Coefficients):")
    
    # Get coefficients
    coef_abs = np.abs(model.coef_)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Abs_Coefficient': coef_abs
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"   Top 10 most influential features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"      {row['Feature']:<25}: {row['Coefficient']:>8.4f} (|{row['Abs_Coefficient']:.4f}|)")
    
    return feature_importance

def generate_week2_summary(feature_importance, test_rmse, test_r2):
    """Generate a summary report for Week 2"""
    print(f"\n" + "=" * 60)
    print("WEEK 2 SUMMARY REPORT")
    print("=" * 60)
    
    print(f"\n📊 FEATURE ENGINEERING RESULTS:")
    print(f"   • Original features: 8")
    print(f"   • New derived features: 10")
    print(f"   • Total features: {len(feature_importance)}")
    
    print(f"\n📈 BASELINE MODEL PERFORMANCE:")
    print(f"   • Test RMSE: {test_rmse:.4f}")
    print(f"   • Test R² Score: {test_r2:.4f}")
    print(f"   • Model successfully trained and evaluated")
    
    print(f"\n🔍 TOP 5 MOST INFLUENTIAL FEATURES:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   • {row['Feature']:<20}: {row['Coefficient']:>8.4f}")
    
    print(f"\n📋 NEXT STEPS FOR WEEK 3:")
    print(f"   1. Train Lasso Regression with hyperparameter tuning")
    print(f"   2. Identify which features are shrunk to zero by Lasso")
    print(f"   3. Plot Lasso coefficient magnitudes")
    print(f"   4. Train Random Forest Regression model")
    print(f"   5. Compare all models' performance")

def main():
    """Main function to run Week 2 tasks"""
    try:
        # Load data
        df = load_data()
        
        # Create derived features
        df, new_features = create_derived_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test, feature_cols = split_data(df)
        
        # Apply scaling
        X_train_scaled, X_test_scaled, scaler = apply_scaling(X_train, X_test)
        
        # Train Linear Regression baseline
        lr_model, y_pred_test, test_rmse, test_r2 = train_linear_regression_baseline(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Analyze feature importance
        feature_importance = analyze_feature_importance(lr_model, feature_cols)
        
        # Generate summary
        generate_week2_summary(feature_importance, test_rmse, test_r2)
        
        # Save processed data and model
        df.to_csv('california_housing_engineered.csv', index=False)
        print(f"\n✅ Processed dataset saved as 'california_housing_engineered.csv'")
        
        print(f"\n🎉 WEEK 2 COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()