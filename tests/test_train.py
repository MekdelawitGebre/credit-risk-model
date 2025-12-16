import pandas as pd
from src.train import train_and_log_models

def test_train_and_log_models_runs(tmp_path):
    df = pd.DataFrame({
        'CustomerId': [1, 2, 3, 4],
        'total_amount': [100, 200, 50, 300],
        'avg_amount': [50, 100, 25, 150],
        'transaction_count': [2, 4, 1, 6],
        'std_amount': [10, 20, 5, 25],
        'recency': [10, 20, 30, 5],
        'frequency': [2, 4, 1, 6],
        'monetary': [100, 200, 50, 300],
        'is_high_risk': [0, 0, 1, 0]
    })
    path = tmp_path / "processed.csv"
    df.to_csv(path, index=False)
    X = df.drop(columns=['is_high_risk', 'CustomerId'])
    y = df['is_high_risk']
    train_and_log_models(X, y)
