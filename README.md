# California Housing Price Prediction Frontend

A web-based frontend for the California Housing Price Prediction project that allows users to:
- Compare performance of Linear Regression, Lasso Regression, and Random Forest models
- Make price predictions using the trained models
- View detailed model performance metrics

## Project Structure

```
├── app.py              # Flask web application
├── main.py             # Main analysis script
├── run_simple_analysis.py  # Simplified analysis script
├── templates/
│   └── index.html      # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css   # Custom styling
│   └── js/
│       └── main.js     # Frontend JavaScript
├── *.pkl               # Trained model files
└── requirements.txt    # Python dependencies
```

## Features

1. **Model Performance Comparison**
   - View RMSE, MAE, and R² metrics for all models across training, validation, and test sets
   - Tabbed interface for easy navigation between different performance metrics

2. **House Price Prediction**
   - Interactive form with all required features
   - Real-time predictions from all three models
   - Visual display of results

3. **Feature Information**
   - Detailed descriptions of all input features
   - Default values based on dataset statistics
   - Input validation with min/max values

## Requirements

- Python 3.7+
- Flask
- Pandas
- NumPy
- Scikit-learn

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`

3. Use the interface to:
   - View model performance metrics
   - Adjust feature values in the input form
   - Click "Predict House Prices" to see predictions from all models

## Model Performance

Based on our analysis:

| Model | Validation RMSE | Test R² |
|-------|----------------|---------|
| Linear Regression | 0.6636 | 0.6469 |
| Lasso Regression | 0.6638 | 0.6468 |
| Random Forest | 0.5227 | 0.7950 |

**Best performing model: Random Forest**

## API Endpoints

- `GET /` - Main interface
- `POST /predict` - Make price predictions
- `GET /api/model_metrics` - Get model performance metrics
- `GET /api/feature_stats` - Get feature statistics

## License

This project is for educational purposes only.