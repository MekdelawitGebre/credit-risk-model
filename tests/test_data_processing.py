import pandas as pd
from src.data_processing import prepare_training_data

def test_pipeline_runs_end_to_end():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2],
        'TransactionId': [10, 11, 20, 21],
        'Amount': [100, 150, 200, 50],
        'TransactionStartTime': ['2025-01-01', '2025-01-02', '2025-02-01', '2025-02-05'],
        'ChannelId': ['web', 'web', 'android', 'android'],
        'ProductCategory': ['electronics', 'electronics', 'fashion', 'fashion']
    })
    features, pipeline = prepare_training_data(df)
    assert features.shape[0] == df['CustomerId'].nunique()
