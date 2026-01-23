#!/usr/bin/env python
"""
Week 1: Exploratory Data Analysis for California Housing Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and perform initial exploration of the California Housing dataset"""
    print("=" * 60)
    print("WEEK 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load the dataset
    print("\n1. Loading California Housing dataset...")
    housing = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    df.rename(columns={'target': 'median_house_value'}, inplace=True)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Basic information
    print(f"\n2. Dataset Information:")
    print(df.info())
    
    # First few rows
    print(f"\n3. First 5 rows:")
    print(df.head())
    
    # Missing values check
    print(f"\n4. Missing Values Check:")
    missing_values = df.isnull().sum()
    print(missing_values)
    if missing_values.sum() == 0:
        print("✅ No missing values found!")
    else:
        print("⚠️  Missing values detected - will need handling")
    
    # Basic statistics
    print(f"\n5. Basic Statistics:")
    print(df.describe())
    
    return df

def analyze_distributions(df):
    """Analyze and visualize feature distributions"""
    print(f"\n6. Feature Distribution Analysis:")
    
    # Create distribution plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(df.columns):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Hide unused subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Distribution plots saved as 'feature_distributions.png'")

def analyze_correlations(df):
    """Analyze feature correlations and multicollinearity"""
    print(f"\n7. Correlation Analysis:")
    
    # Create correlation matrix
    corr_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
    plt.title('Correlation Matrix of California Housing Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")
    
    # Identify highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    print(f"\nHighly correlated feature pairs (|r| > 0.7):")
    if high_corr_pairs:
        for pair in high_corr_pairs:
            print(f"   {pair[0]} - {pair[1]}: {pair[2]:.3f}")
    else:
        print("   No highly correlated pairs found (|r| > 0.7)")
    
    return corr_matrix

def calculate_vif(df):
    """Calculate Variance Inflation Factor for multicollinearity detection"""
    print(f"\n8. Variance Inflation Factor (VIF) Analysis:")
    
    def calculate_vif_single(df, features, target_feature):
        """Calculate VIF for a single feature"""
        X = df[features].drop(target_feature, axis=1)
        y = df[target_feature]
        
        # Fit the model
        reg = LinearRegression()
        reg.fit(X, y)
        r_squared = reg.score(X, y)
        
        # Calculate VIF
        if r_squared >= 1:
            return float('inf')  # Perfect multicollinearity
        return 1 / (1 - r_squared)
    
    # Calculate VIF for all features except target
    features_for_vif = [col for col in df.columns if col != 'median_house_value']
    vif_results = []
    
    for feature in features_for_vif:
        try:
            vif = calculate_vif_single(df, features_for_vif, feature)
            vif_results.append({'Feature': feature, 'VIF': vif})
        except:
            vif_results.append({'Feature': feature, 'VIF': float('inf')})
    
    vif_df = pd.DataFrame(vif_results)
    vif_df = vif_df.sort_values('VIF', ascending=False)
    
    print("Variance Inflation Factors:")
    print(vif_df)
    
    # Identify features with high VIF
    high_vif_features = vif_df[vif_df['VIF'] > 5]['Feature'].tolist()
    print(f"\nFeatures with high VIF (>5): {high_vif_features}")
    
    if len(high_vif_features) > 0:
        print("⚠️  High multicollinearity detected - consider feature selection or dimensionality reduction")
    else:
        print("✅ Low multicollinearity - features are relatively independent")
    
    return vif_df

def generate_summary_report(df, corr_matrix, vif_df):
    """Generate a summary report of Week 1 findings"""
    print("\n" + "=" * 60)
    print("WEEK 1 SUMMARY REPORT")
    print("=" * 60)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Features: {len(df.columns) - 1} (excluding target)")
    print(f"   • Target variable: median_house_value")
    print(f"   • Missing values: {df.isnull().sum().sum()} (None found)")
    
    print(f"\n📈 TARGET VARIABLE ANALYSIS:")
    target_stats = df['median_house_value'].describe()
    print(f"   • Mean: ${target_stats['mean']*100000:.0f}")
    print(f"   • Std Dev: ${target_stats['std']*100000:.0f}")
    print(f"   • Min: ${target_stats['min']*100000:.0f}")
    print(f"   • Max: ${target_stats['max']*100000:.0f}")
    
    print(f"\n🔍 KEY FINDINGS:")
    
    # Most correlated features with target
    target_corr = corr_matrix['median_house_value'].abs().sort_values(ascending=False)
    top_features = target_corr[1:4]  # Exclude target itself
    print(f"   • Top 3 features correlated with house prices:")
    for feature, corr_val in top_features.items():
        print(f"     - {feature}: {corr_val:.3f}")
    
    # Multicollinearity summary
    high_vif_count = len(vif_df[vif_df['VIF'] > 5])
    if high_vif_count > 0:
        print(f"   • Multicollinearity: {high_vif_count} features show high VIF (>5)")
    else:
        print(f"   • Multicollinearity: Low - all features are relatively independent")
    
    print(f"\n📋 NEXT STEPS FOR WEEK 2:")
    print(f"   1. Feature engineering (derived features)")
    print(f"   2. Train-test split")
    print(f"   3. Feature scaling")
    print(f"   4. Build Linear Regression baseline model")

def main():
    """Main function to run Week 1 analysis"""
    try:
        # Load and explore data
        df = load_and_explore_data()
        
        # Analyze distributions
        analyze_distributions(df)
        
        # Analyze correlations
        corr_matrix = analyze_correlations(df)
        
        # Calculate VIF
        vif_df = calculate_vif(df)
        
        # Generate summary report
        generate_summary_report(df, corr_matrix, vif_df)
        
        # Save cleaned data for next week
        df.to_csv('california_housing_cleaned.csv', index=False)
        print(f"\n✅ Cleaned dataset saved as 'california_housing_cleaned.csv'")
        
        print("\n🎉 WEEK 1 ANALYSIS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()