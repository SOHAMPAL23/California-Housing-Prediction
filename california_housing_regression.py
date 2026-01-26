#!/usr/bin/env python
"""
California Housing Price Prediction (Regression)
🎯 Objective: Build, evaluate, and compare Linear Regression, Lasso Regression, and Random Forest Regression models
to predict median house values in California using tabular housing features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """1️⃣ Exploratory Data Analysis (EDA)"""
    print("="*60)
    print("1️⃣ EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    # Load the California Housing dataset
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    df.rename(columns={'target': 'median_house_value'}, inplace=True)
    
    # Display basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    
    # Display first few rows
    print(f"\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values in each column:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    # Check for basic statistics
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Visualize feature distributions
    print(f"\nCreating feature distribution visualizations...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(df.columns):
        if col != 'median_house_value':
            axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
    
    # Hide the last subplot (since we have 8 features + 1 target = 9 total, but only 8 features to plot)
    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create correlation heatmap
    print(f"\nCreating correlation heatmap...")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
    plt.title('Correlation Matrix of California Housing Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Check multicollinearity using VIF (Variance Inflation Factor)
    print(f"\nChecking multicollinearity using correlation matrix...")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    print('Highly correlated feature pairs (|r| > 0.7):')
    for pair in high_corr_pairs:
        print(f'{pair[0]} - {pair[1]}: {pair[2]:.3f}')
    
    return df, housing.feature_names

def feature_engineering_and_scaling(df, feature_names):
    """2️⃣ Feature Engineering & Scaling"""
    print("\n" + "="*60)
    print("2️⃣ FEATURE ENGINEERING & SCALING")
    print("="*60)
    
    # Create derived features
    print("Creating derived features...")
    df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
    df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
    df['population_per_household'] = df['Population'] / df['AveOccup']
    
    # Define features and target
    feature_cols = [col for col in df.columns if col != 'median_house_value']
    X = df[feature_cols]
    y = df['median_house_value']
    
    # Split data into train/test sets
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply feature scaling (required for Linear Regression and Lasso)
    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def train_linear_regression(X_train, X_test, y_train, y_test, feature_cols):
    """3️⃣ Model 1: Linear Regression (Baseline)"""
    print("\n" + "="*60)
    print("3️⃣ MODEL 1: LINEAR REGRESSION (BASELINE)")
    print("="*60)
    
    # Train Linear Regression model
    print("Training Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    # Evaluate using RMSE and R²
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Linear Regression Results:")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    # Plot Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression: Predicted vs Actual Values')
    plt.tight_layout()
    plt.savefig('linear_regression_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Residuals vs Predicted plot
    residuals = y_test - y_pred_test
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_test, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Linear Regression: Residuals vs Predicted')
    plt.tight_layout()
    plt.savefig('linear_regression_residuals_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Interpret coefficients to understand feature influence
    print(f"\nFeature Coefficients (Linear Regression):")
    coefficients = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': lr_model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    
    print(coefficients)
    
    return lr_model, test_rmse, test_r2, coefficients

def train_lasso_regression(X_train, X_test, y_train, y_test, feature_cols):
    """4️⃣ Model 2: Lasso Regression (Feature Selection)"""
    print("\n" + "="*60)
    print("4️⃣ MODEL 2: LASSO REGRESSION (FEATURE SELECTION)")
    print("="*60)
    
    # Train Lasso with hyperparameter tuning (alpha)
    print("Training Lasso with hyperparameter tuning...")
    alphas = np.logspace(-4, 1, 50)  # Range of alpha values to try
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
    lasso_cv.fit(X_train, y_train)
    
    # Train final model with best alpha
    best_alpha = lasso_cv.alpha_
    print(f"Best alpha found: {best_alpha}")
    
    lasso_model = Lasso(alpha=best_alpha, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_test = lasso_model.predict(X_test)
    
    # Evaluate using RMSE and R²
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Lasso Regression Results (alpha={best_alpha}):")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    # Identify which features are shrunk to zero
    zero_coef_features = [feature_cols[i] for i, coef in enumerate(lasso_model.coef_) if coef == 0]
    nonzero_coef_features = [feature_cols[i] for i, coef in enumerate(lasso_model.coef_) if coef != 0]
    
    print(f"\nFeatures shrunk to zero by Lasso: {len(zero_coef_features)}")
    for feat in zero_coef_features:
        print(f"  - {feat}")
    
    print(f"\nFeatures retained by Lasso: {len(nonzero_coef_features)}")
    for feat in nonzero_coef_features:
        print(f"  - {feat}")
    
    # Compare performance vs Linear Regression
    print(f"\nComparison with Linear Regression:")
    print(f"  Lasso Test R²: {test_r2:.4f}")
    
    # Plot coefficient magnitudes
    plt.figure(figsize=(12, 6))
    coefs = lasso_model.coef_
    plt.bar(range(len(coefs)), coefs)
    plt.xticks(range(len(coefs)), feature_cols, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title(f'Lasso Coefficient Magnitudes (alpha={best_alpha})')
    plt.tight_layout()
    plt.savefig('lasso_coefficient_magnitudes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return lasso_model, test_rmse, test_r2, best_alpha

def train_random_forest(X_train, X_test, y_train, y_test, feature_cols):
    """5️⃣ Model 3: Random Forest Regression"""
    print("\n" + "="*60)
    print("5️⃣ MODEL 3: RANDOM FOREST REGRESSION")
    print("="*60)
    
    # Train Random Forest with reasonable hyperparameters
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_test = rf_model.predict(X_test)
    
    # Evaluate using RMSE and R²
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Random Forest Results:")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    # Feature importance rankings
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print(f"\nFeature Importance (Random Forest):")
    print(feature_importance)
    
    # Plot feature importance rankings
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(rf_model.feature_importances_)), rf_model.feature_importances_)
    plt.xticks(range(len(rf_model.feature_importances_)), feature_cols, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_model, test_rmse, test_r2, feature_importance

def compare_models(lr_results, lasso_results, rf_results, lr_coefficients, rf_importance):
    """Compare all models and summarize results"""
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # Create comparison table
    comparison_data = {
        'Model': ['Linear Regression', 'Lasso', 'Random Forest'],
        'Test_RMSE': [lr_results[0], lasso_results[0], rf_results[0]],
        'Test_R2': [lr_results[1], lasso_results[1], rf_results[1]]
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    print("Model Comparison Table:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Identify best performing model
    best_r2_idx = comparison_df['Test_R2'].idxmax()
    best_model = comparison_df.loc[best_r2_idx, 'Model']
    print(f"\nBest performing model based on R²: {best_model}")
    
    # Analyze interpretability vs accuracy trade-offs
    print(f"\n🔍 INTERPRETABILITY VS ACCURACY TRADE-OFFS:")
    print(f"Linear Regression:")
    print(f"  - Accuracy: Moderate")
    print(f"  - Interpretability: High (direct coefficient interpretation)")
    print(f"  - Feature Selection: Manual")
    
    print(f"\nLasso Regression:")
    print(f"  - Accuracy: Moderate to Good")
    print(f"  - Interpretability: High (coefficients + feature selection)")
    print(f"  - Feature Selection: Automatic (shrinks some coefficients to zero)")
    
    print(f"\nRandom Forest:")
    print(f"  - Accuracy: High")
    print(f"  - Interpretability: Lower (ensemble of trees)")
    print(f"  - Feature Selection: Implicit (feature importance)")

def answer_key_questions(df, lr_coefficients, rf_importance):
    """Answer the key questions from the problem statement"""
    print("\n" + "="*60)
    print("❓ ANSWERING KEY QUESTIONS")
    print("="*60)
    
    print("1. Which features most strongly drive housing prices?")
    print("   Based on Linear Regression (magnitude of coefficients):")
    lr_top_features = lr_coefficients.head(5)
    for idx, row in lr_top_features.iterrows():
        print(f"      {row['Feature']}: {row['Coefficient']:.4f}")
    
    print("\n   Based on Random Forest (feature importance):")
    rf_top_features = rf_importance.head(5)
    for idx, row in rf_top_features.iterrows():
        print(f"      {row['Feature']}: {row['Importance']:.4f}")
    
    print("\n2. How do linear and non-linear models compare in performance?")
    print("   Linear models (Linear Regression, Lasso) provide interpretable results")
    print("   Non-linear models (Random Forest) typically achieve higher accuracy")
    print("   The trade-off is between interpretability and performance")
    
    print("\n3. What interpretability is gained or lost when moving from linear to ensemble models?")
    print("   Linear models: Direct coefficient interpretation, clear feature relationships")
    print("   Ensemble models: Higher accuracy, feature importance scores, but less direct interpretation")
    print("   Lasso: Balances interpretability with automatic feature selection")

def main():
    """Main function to run the complete California Housing Price Prediction project"""
    print("🏠 CALIFORNIA HOUSING PRICE PREDICTION PROJECT")
    print("🎯 Objective: Build, evaluate, and compare Linear Regression, Lasso Regression,")
    print("   and Random Forest Regression models to predict median house values in California")
    
    # 1️⃣ Exploratory Data Analysis (EDA)
    df, feature_names = load_and_explore_data()
    
    # 2️⃣ Feature Engineering & Scaling
    X_train, X_test, y_train, y_test, scaler, feature_cols = feature_engineering_and_scaling(df, feature_names)
    
    # 3️⃣ Model 1: Linear Regression (Baseline)
    lr_model, lr_rmse, lr_r2, lr_coefficients = train_linear_regression(
        X_train, X_test, y_train, y_test, feature_cols
    )
    
    # 4️⃣ Model 2: Lasso Regression (Feature Selection)
    lasso_model, lasso_rmse, lasso_r2, best_alpha = train_lasso_regression(
        X_train, X_test, y_train, y_test, feature_cols
    )
    
    # 5️⃣ Model 3: Random Forest Regression
    rf_model, rf_rmse, rf_r2, rf_importance = train_random_forest(
        X_train, X_test, y_train, y_test, feature_cols
    )
    
    # Compare all models
    compare_models(
        (lr_rmse, lr_r2), 
        (lasso_rmse, lasso_r2), 
        (rf_rmse, rf_r2),
        lr_coefficients,
        rf_importance
    )
    
    # Answer key questions
    answer_key_questions(df, lr_coefficients, rf_importance)
    
    print("\n" + "="*60)
    print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("📊 All required visualizations saved as PNG files")
    print("📈 Model comparison and insights provided")
    print("🔍 Key questions answered")
    print("="*60)

if __name__ == "__main__":
    main()