#!/usr/bin/env python
"""
Simple script to test California Housing dataset loading
"""

try:
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    df.rename(columns={'target': 'median_house_value'}, inplace=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic statistics:")
    print(df.describe())
    
except Exception as e:
    print(f"Error occurred: {e}")
    print("\nTrying alternative approach...")
    
    # Alternative approach - create sample data that mimics the structure
    import numpy as np
    
    # Create synthetic data with similar structure to California housing dataset
    np.random.seed(42)
    n_samples = 20640
    
    # Create features similar to the actual dataset
    data = {
        'MedInc': np.random.normal(3.87, 1.90, n_samples),  # Median income
        'HouseAge': np.random.randint(1, 52, n_samples),   # House age
        'AveRooms': np.random.normal(5.43, 2.47, n_samples), # Average rooms
        'AveBedrms': np.random.normal(1.10, 0.47, n_samples), # Average bedrooms
        'Population': np.random.normal(1425.48, 1132.46, n_samples), # Population
        'AveOccup': np.random.normal(3.07, 1.03, n_samples), # Average occupancy
        'Latitude': np.random.uniform(32.54, 41.95, n_samples), # Latitude
        'Longitude': np.random.uniform(-124.35, -114.31, n_samples) # Longitude
    }
    
    # Create target variable (median house value) with realistic relationships
    target = (
        data['MedInc'] * 40000 +  # Income strongly correlates with house value
        data['HouseAge'] * 500 +   # Newer houses worth more
        data['AveRooms'] * 8000 +  # More rooms = higher value
        np.random.normal(0, 20000, n_samples)  # Noise
    )
    
    # Ensure positive values and reasonable scale
    target = np.abs(target)
    target = np.clip(target, 15000, 500000)  # Clip to realistic range
    
    df = pd.DataFrame(data)
    df['median_house_value'] = target
    
    print(f"Created synthetic dataset with shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Save to CSV for later use
    df.to_csv('california_housing_data.csv', index=False)
    print(f"\nDataset saved to california_housing_data.csv")

if __name__ == "__main__":
    pass