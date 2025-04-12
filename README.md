# Payment Fraud Detection System

This project implements a machine learning-based payment fraud detection system using various algorithms and techniques.

## Project Structure

```
fraud_detection/
├── data/               # Dataset directory
├── docs/              # Documentation
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
└── tests/            # Unit tests
```

## Features

- Dataset with 50,000 transactions and 10 features
- 8 different types of visualizations for data analysis
- Implementation of 4 different ML models:
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- Comprehensive documentation and analysis

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python src/main.py
```

## Data Visualization

The project includes the following visualizations:
1. Transaction Amount Distribution
2. Fraud vs Non-Fraud Distribution
3. Correlation Heatmap
4. Feature Importance Plot
5. ROC Curves
6. Confusion Matrices
7. Transaction Time Analysis
8. Geographic Distribution of Fraud

## Models

The following machine learning models are implemented:
1. Random Forest Classifier
2. XGBoost Classifier
3. LightGBM Classifier
4. CatBoost Classifier

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

## Results

Detailed model performance metrics and comparisons can be found in the notebooks directory.

## Contributing

Feel free to submit issues and enhancement requests. 