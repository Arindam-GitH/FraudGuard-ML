# Payment Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit%20Learn%20%7C%20XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project for detecting fraudulent payment transactions. Features 50,000 synthetic transactions, 4 ML models (Random Forest, XGBoost, LightGBM, CatBoost), 8 data visualizations, and SMOTE for handling class imbalance. Perfect for financial institutions and data scientists working on fraud detection systems.

## 🚀 Features

- **Dataset Generation**: 50,000 synthetic transactions with 10 relevant features
- **Multiple ML Models**: Implementation of 4 powerful algorithms
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- **Advanced Visualizations**: 8 different types of data visualizations
- **Class Imbalance Handling**: SMOTE technique implementation
- **Comprehensive Evaluation**: Multiple performance metrics

## 📊 Project Structure

```
fraud_detection/
├── data/               # Dataset directory
├── docs/              # Documentation and visualizations
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
└── tests/            # Unit tests
```

## 🛠️ Technical Stack

- **Languages**: Python 3.x
- **ML Libraries**: 
  - Scikit-learn
  - XGBoost
  - LightGBM
  - CatBoost
- **Data Processing**: 
  - Pandas
  - NumPy
- **Visualization**: 
  - Matplotlib
  - Seaborn
  - Plotly
- **Development**: 
  - Jupyter Notebooks
  - Git

## 🚀 Getting Started

1. **Clone the repository**
```bash
git clone https://github.com/Arindam-GitH/Payment-Fraud-Detection-.git
cd Payment-Fraud-Detection-
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the analysis**
```bash
python src/main.py
```

## 📈 Model Performance

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

## 📊 Visualizations

The project includes 8 different types of visualizations:
1. Transaction Amount Distribution
2. Fraud vs Non-Fraud Distribution
3. Correlation Heatmap
4. Transaction Time Analysis
5. Amount vs Fraud Box Plot
6. Customer Age Distribution
7. Device Type Distribution
8. Geographic Distribution

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Arindam Guha**
- GitHub: [@Arindam-GitH](https://github.com/Arindam-GitH)

---
⭐️ From [Arindam-GitH](https://github.com/Arindam-GitH) 