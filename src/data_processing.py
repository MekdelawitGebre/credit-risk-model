import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE
from sklearn.base import BaseEstimator, TransformerMixin


def compute_rfm(df: pd.DataFrame):
    """Compute Recency, Frequency, Monetary features per Customer."""    df = df.copy()
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
class RFMTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to compute Recency, Frequency, Monetary metrics."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        snapshot_date = X['TransactionStartTime'].max() + pd.Timedelta(days=1)
        rfm = X.groupby('CustomerId').agg(
            recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
            frequency=('TransactionId', 'count'),
            monetary=('Amount', 'sum')
        ).reset_index()
        return rfm


def build_feature_pipeline(categorical_features, numeric_features):
    """Build a complete sklearn pipeline with imputation, encoding, and scaling."""

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    return pipeline


def compute_woe_iv(df: pd.DataFrame, target_col: str):
    """Compute Weight of Evidence and Information Value using xverse."""
    woe = WOE()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if not cat_cols:
        return pd.DataFrame()
    woe.fit(df[cat_cols], df[target_col])
    _ = woe.transform(df[cat_cols])
    iv_dict = woe.iv_dict
    iv_df = pd.DataFrame(list(iv_dict.items()), columns=['Feature', 'IV'])
    return iv_df


def prepare_training_data(df: pd.DataFrame):
    """End-to-end data preprocessing for model training."""
    rfm = RFMTransformer().transform(df)

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
    merged = agg.merge(rfm, on='CustomerId', how='left')

    categorical_features = ['channel', 'product_category']
    numeric_features = [
        'total_amount', 'avg_amount', 'transaction_count',
        'std_amount', 'recency', 'frequency', 'monetary'
    ]

    pipeline = build_feature_pipeline(categorical_features, numeric_features)
    processed = pipeline.fit_transform(merged)

    return processed, pipeline
