import pandas as pd
from src.data_processing import create_processed_dataset

def test_proxy_target_creation():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 3, 3],
        'TransactionId': [10, 11, 20, 21, 30, 31],
        'Amount': [100, 150, 50, 70, 20, 30],
        'TransactionStartTime': [
            '2025-01-01', '2025-01-02',
            '2025-02-01', '2025-02-02',
            '2025-03-01', '2025-03-02'
        ],
        'ChannelId': ['web', 'web', 'android', 'android', 'ios', 'ios'],
        'ProductCategory': ['electronics', 'electronics', 'fashion', 'fashion', 'groceries', 'groceries']
    })
    processed = create_processed_dataset(df)
    assert 'is_high_risk' in processed.columns
    assert processed['is_high_risk'].isin([0, 1]).all()
