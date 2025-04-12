import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go

def generate_synthetic_data(n_samples=50000):
    """Generate synthetic payment transaction data."""
    np.random.seed(42)
    
    # Generate features
    data = {
        'transaction_id': range(n_samples),
        'amount': np.random.exponential(100, n_samples),  # Most transactions are small
        'time': np.random.uniform(0, 24, n_samples),  # Hour of day
        'merchant_category': np.random.randint(1, 10, n_samples),
        'customer_age': np.random.normal(35, 10, n_samples),
        'customer_income': np.random.normal(50000, 20000, n_samples),
        'distance_from_home': np.random.exponential(10, n_samples),
        'previous_transactions': np.random.poisson(5, n_samples),
        'device_type': np.random.randint(1, 4, n_samples),  # 1: Mobile, 2: Desktop, 3: Tablet
        'location_mismatch': np.random.binomial(1, 0.1, n_samples)  # 1 if location doesn't match usual pattern
    }
    
    df = pd.DataFrame(data)
    
    # Generate fraud labels (about 1% of transactions are fraudulent)
    fraud_prob = 0.01
    df['is_fraud'] = np.random.binomial(1, fraud_prob, n_samples)
    
    # Adjust fraud probability based on features
    df.loc[df['amount'] > df['amount'].quantile(0.95), 'is_fraud'] = np.random.binomial(1, 0.1, len(df[df['amount'] > df['amount'].quantile(0.95)]))
    df.loc[df['location_mismatch'] == 1, 'is_fraud'] = np.random.binomial(1, 0.2, len(df[df['location_mismatch'] == 1]))
    
    return df

def create_visualizations(df):
    """Create 8 different visualizations for the dataset."""
    # 1. Transaction Amount Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='amount', hue='is_fraud', bins=50)
    plt.title('Transaction Amount Distribution by Fraud Status')
    plt.savefig('docs/amount_distribution.png')
    plt.close()

    # 2. Fraud vs Non-Fraud Distribution
    plt.figure(figsize=(8, 6))
    df['is_fraud'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Fraud vs Non-Fraud Transactions')
    plt.savefig('docs/fraud_distribution.png')
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('docs/correlation_heatmap.png')
    plt.close()

    # 4. Transaction Time Analysis
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='time', hue='is_fraud', bins=24)
    plt.title('Transaction Time Distribution by Fraud Status')
    plt.savefig('docs/time_distribution.png')
    plt.close()

    # 5. Amount vs Fraud (Box Plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='is_fraud', y='amount')
    plt.title('Transaction Amount Distribution by Fraud Status')
    plt.savefig('docs/amount_boxplot.png')
    plt.close()

    # 6. Customer Age vs Fraud
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='customer_age', hue='is_fraud', bins=30)
    plt.title('Customer Age Distribution by Fraud Status')
    plt.savefig('docs/age_distribution.png')
    plt.close()

    # 7. Device Type vs Fraud
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='device_type', hue='is_fraud')
    plt.title('Device Type Distribution by Fraud Status')
    plt.savefig('docs/device_distribution.png')
    plt.close()

    # 8. Geographic Distribution (using distance_from_home as proxy)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='distance_from_home', y='amount', hue='is_fraud', alpha=0.5)
    plt.title('Transaction Amount vs Distance from Home')
    plt.savefig('docs/geographic_distribution.png')
    plt.close()

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple ML models."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_curve(y_test, y_pred_proba)
        }
        
        # Save model
        model_path = f'models/{name.lower().replace(" ", "_")}.pkl'
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    return results

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    # Save dataset
    df.to_csv('data/transactions.csv', index=False)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df)
    
    # Prepare data for modeling
    X = df.drop(['transaction_id', 'is_fraud'], axis=1)
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train and evaluate models
    print("Training and evaluating models...")
    results = train_and_evaluate_models(X_train_balanced, X_test, y_train_balanced, y_test)
    
    # Print results
    print("\nModel Performance Summary:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print(model_results['classification_report'])

if __name__ == "__main__":
    main() 