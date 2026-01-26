#!/usr/bin/env python
"""
Week 3: Lasso Regression and Random Forest Models for California Housing Prediction
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for modeling"""
    print("=" * 60)
    print("WEEK 3: LASSO REGRESSION AND RANDOM FOREST MODELS")
    print("=" * 60)
    
    # Load the dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    df.rename(columns={'target': 'median_house_value'}, inplace=True)
    
    # Create derived features (similar to week 2)
    df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
    df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
    df['population_per_household'] = df['Population'] / df['AveOccup']
    df['rooms_per_person'] = df['AveRooms'] / df['AveOccup']
    df['bedrooms_per_person'] = df['AveBedrms'] / df['AveOccup']
    df['household_size'] = df['Population'] / df['AveOccup']
    df['lat_abs'] = df['Latitude'].abs()
    df['long_abs'] = df['Longitude'].abs()
    df['income_per_person'] = df['MedInc'] / df['AveOccup']
    df['pop_density_proxy'] = df['Population'] / (df['AveOccup'] * df['HouseAge'])
    
    # Define features and target
    feature_cols = [col for col in df.columns if col != 'median_house_value']
    X = df[feature_cols]
    y = df['median_house_value']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data prepared with {X_train.shape[1]} features")
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

def train_lasso_with_cv(X_train, y_train, X_test, y_test):
    """Train Lasso Regression with cross-validation for hyperparameter tuning"""
    print(f"\n1. Training Lasso Regression with Hyperparameter Tuning:")
    
    # Use LassoCV to find optimal alpha through cross-validation
    alphas = np.logspace(-4, 1, 50)  # Range of alpha values to try
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
    lasso_cv.fit(X_train, y_train)
    
    # Get the best alpha
    best_alpha = lasso_cv.alpha_
    print(f"   Best alpha found: {best_alpha:.6f}")
    
    # Train final model with best alpha
    lasso_model = Lasso(alpha=best_alpha, random_state=42, max_iter=2000)
    lasso_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = lasso_model.predict(X_train)
    y_pred_test = lasso_model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"   ✅ Lasso Regression trained successfully")
    print(f"   📊 Training Metrics:")
    print(f"      - RMSE: {train_rmse:.4f}")
    print(f"      - R² Score: {train_r2:.4f}")
    print(f"   📊 Test Metrics:")
    print(f"      - RMSE: {test_rmse:.4f}")
    print(f"      - R² Score: {test_r2:.4f}")
    
    return lasso_model, y_pred_test, test_rmse, test_r2, best_alpha

def identify_zero_coefficients(lasso_model, feature_names):
    """Identify which features are shrunk to zero by Lasso"""
    print(f"\n2. Identifying Features Shrunk to Zero by Lasso:")
    
    # Get coefficients
    coef = lasso_model.coef_
    
    # Find features with zero coefficients
    zero_coef_indices = np.where(coef == 0)[0]
    zero_coef_features = [feature_names[i] for i in zero_coef_indices]
    
    # Find features with non-zero coefficients
    non_zero_coef_indices = np.where(coef != 0)[0]
    non_zero_coef_features = [feature_names[i] for i in non_zero_coef_indices]
    
    print(f"   ✅ Lasso feature selection results:")
    print(f"      - Features removed (coefficient = 0): {len(zero_coef_features)}")
    print(f"      - Features retained: {len(non_zero_coef_features)}")
    
    if zero_coef_features:
        print(f"      - Removed features:")
        for feature in zero_coef_features[:10]:  # Show first 10
            print(f"        * {feature}")
        if len(zero_coef_features) > 10:
            print(f"        * ... and {len(zero_coef_features)-10} more")
    
    print(f"      - Retained features (top 10 by absolute coefficient):")
    # Get non-zero coefficients with their features
    non_zero_coefs = [(feature_names[i], coef[i]) for i in non_zero_coef_indices]
    non_zero_coefs_sorted = sorted(non_zero_coefs, key=lambda x: abs(x[1]), reverse=True)
    
    for feature, coef_val in non_zero_coefs_sorted[:10]:
        print(f"        * {feature:<25}: {coef_val:>8.4f}")
    
    return zero_coef_features, non_zero_coef_features

def plot_lasso_coefficients(lasso_model, feature_names):
    """Plot Lasso coefficient magnitudes"""
    print(f"\n3. Preparing Lasso Coefficient Visualization:")
    
    # Get coefficients
    coef = lasso_model.coef_
    
    # Create dataframe for visualization
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Abs_Coefficient': np.abs(coef)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Filter to top 15 features for better visualization
    top_15 = coef_df.head(15)
    
    print(f"   ✅ Coefficient data prepared")
    print(f"      - Top 15 features by coefficient magnitude:")
    for _, row in top_15.iterrows():
        print(f"        {row['Feature']:<25}: {row['Coefficient']:>8.4f}")
    
    return coef_df

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest Regression model"""
    print(f"\n4. Training Random Forest Regression Model:")
    
    # Train Random Forest with reasonable parameters
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"   ✅ Random Forest trained successfully")
    print(f"   📊 Training Metrics:")
    print(f"      - RMSE: {train_rmse:.4f}")
    print(f"      - R² Score: {train_r2:.4f}")
    print(f"   📊 Test Metrics:")
    print(f"      - RMSE: {test_rmse:.4f}")
    print(f"      - R² Score: {test_r2:.4f}")
    
    return rf_model, y_pred_test, test_rmse, test_r2

def compare_models(lasso_results, rf_results, lasso_test_r2, rf_test_r2):
    """Compare model performances"""
    print(f"\n5. Model Comparison:")
    
    print(f"   📊 Performance Comparison:")
    print(f"      Model          | Test RMSE | Test R² Score")
    print(f"      ---------------|-----------|------------")
    print(f"      Lasso          | {lasso_results[2]:>9.4f} | {lasso_test_r2:>11.4f}")
    print(f"      Random Forest  | {rf_results[2]:>9.4f} | {rf_test_r2:>11.4f}")
    
    # Determine best model
    if rf_test_r2 > lasso_test_r2:
        print(f"   🏆 Random Forest performs better based on R² score")
        better_model = "Random Forest"
    elif lasso_test_r2 > rf_test_r2:
        print(f"   🏆 Lasso performs better based on R² score")
        better_model = "Lasso"
    else:
        print(f"   🤝 Both models perform equally")
        better_model = "Both"
    
    return better_model

def generate_week3_summary(lasso_model, rf_model, feature_names, zero_coef_features, 
                          non_zero_coef_features, coef_df, better_model):
    """Generate a summary report for Week 3"""
    print(f"\n" + "=" * 60)
    print("WEEK 3 SUMMARY REPORT")
    print("=" * 60)
    
    print(f"\n📊 LASSO REGRESSION RESULTS:")
    print(f"   • Best alpha: {lasso_model.alpha_:.6f}")
    print(f"   • Features removed by Lasso: {len(zero_coef_features)}")
    print(f"   • Features retained: {len(non_zero_coef_features)}")
    print(f"   • Sparsity: {(len(zero_coef_features)/len(feature_names))*100:.1f}%")
    
    print(f"\n🌳 RANDOM FOREST RESULTS:")
    print(f"   • Number of trees: {rf_model.n_estimators}")
    print(f"   • Max depth: {rf_model.max_depth}")
    
    print(f"\n🔍 FEATURE SELECTION INSIGHTS:")
    print(f"   • Lasso achieved feature selection by shrinking coefficients to zero")
    print(f"   • Most important features identified by Lasso (top 5):")
    top_5_lasso = coef_df[coef_df['Coefficient'] != 0].head(5)
    for _, row in top_5_lasso.iterrows():
        print(f"     - {row['Feature']}: {row['Coefficient']:.4f}")
    
    print(f"\n🏆 BEST PERFORMING MODEL: {better_model}")
    
    print(f"\n📋 NEXT STEPS FOR WEEK 4:")
    print(f"   1. Create Random Forest feature importance plot")
    print(f"   2. Summarize results in comparison table for all models")
    print(f"   3. Analyze what drives house prices most based on all models")
    print(f"   4. Create final notebook with clean code, markdown explanations and visualizations")

def main():
    """Main function to run Week 3 tasks"""
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare_data()
        
        # Train Lasso with hyperparameter tuning
        lasso_model, lasso_pred, lasso_rmse, lasso_r2, best_alpha = train_lasso_with_cv(
            X_train, y_train, X_test, y_test
        )
        
        # Identify features shrunk to zero
        zero_coef_features, non_zero_coef_features = identify_zero_coefficients(
            lasso_model, feature_names
        )
        
        # Plot Lasso coefficients
        coef_df = plot_lasso_coefficients(lasso_model, feature_names)
        
        # Train Random Forest
        rf_model, rf_pred, rf_rmse, rf_r2 = train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        # Compare models
        better_model = compare_models(
            (X_train, X_test, lasso_rmse, lasso_r2), 
            (X_train, X_test, rf_rmse, rf_r2),
            lasso_r2, rf_r2
        )
        
        # Generate summary
        generate_week3_summary(
            lasso_model, rf_model, feature_names, zero_coef_features, 
            non_zero_coef_features, coef_df, better_model
        )
        
        print(f"\n🎉 WEEK 3 COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()