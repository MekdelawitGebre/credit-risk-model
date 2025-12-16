import pandas as pd
from src.data_processing import preprocess_transactions

def test_preprocess_returns_expected_columns():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionId": [10, 11, 12],
        "Amount": [100, 200, 50],
        "TransactionStartTime": ["2025-01-01", "2025-01-02", "2025-02-01"]
    })
    result = preprocess_transactions(df)
    expected_cols = {'CustomerId', 'total_amount', 'avg_amount', 'transaction_count', 'std_amount'}
    assert set(result.columns).issuperset(expected_cols)
