import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def compute_rfm(df: pd.DataFrame):
    """Compute Recency, Frequency, Monetary features per Customer."""
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg(
        recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        frequency=('TransactionId', 'count'),
        monetary=('Amount', 'sum')
    ).reset_index()
    return rfm


def cluster_customers(rfm: pd.DataFrame, n_clusters: int = 3):
    """Cluster customers based on RFM metrics using K-Means and assign high-risk labels."""
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_rfm)
    rfm['cluster'] = clusters

    # Identify the high-risk cluster (lowest monetary/frequency)
    cluster_summary = (
        rfm.groupby('cluster')[['monetary', 'frequency', 'recency']]
        .mean()
        .reset_index()
    )

    high_risk_cluster = cluster_summary.sort_values(by=['monetary', 'frequency']).iloc[0]['cluster']
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

    return rfm[['CustomerId', 'is_high_risk', 'cluster']]


def create_processed_dataset(df: pd.DataFrame):
    """Create the processed dataset with RFM-based risk labels."""
    # --- Feature engineering ---
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    agg = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_amount=('Amount', 'std'),
        channel=('ChannelId', 'first'),
        product_category=('ProductCategory', 'first')
    ).reset_index()

    # --- Compute RFM and cluster customers ---
    rfm = compute_rfm(df)
    risk_labels = cluster_customers(rfm, n_clusters=3)

    # --- Merge RFM risk labels into processed dataset ---
    processed = agg.merge(risk_labels, on='CustomerId', how='left')
    processed.fillna({'is_high_risk': 0}, inplace=True)

    return processed
