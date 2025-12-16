import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['hour'] = df['TransactionStartTime'].dt.hour
    df['day'] = df['TransactionStartTime'].dt.day
    df['month'] = df['TransactionStartTime'].dt.month
    df['year'] = df['TransactionStartTime'].dt.year
    df['Value'] = df['Value'].abs()

    agg = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_amount=('Amount', 'std')
    ).reset_index()
    return agg


def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg(
        recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        frequency=('TransactionId', 'count'),
        monetary=('Amount', 'sum')
    )
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    km = KMeans(n_clusters=3, random_state=42)
    rfm['cluster'] = km.fit_predict(rfm_scaled)
    rfm['is_high_risk'] = (rfm['cluster'] == rfm.groupby('cluster')['monetary'].mean().idxmin()).astype(int)
    return rfm[['is_high_risk']]

def create_processed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    features = preprocess_transactions(df)
    target = rfm_segmentation(df)
    merged = features.merge(target, on='CustomerId', how='left')
    merged.fillna(0, inplace=True)
    return merged
