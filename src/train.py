import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import joblib

def train_models(df: pd.DataFrame):
    X = df.drop(columns=['is_high_risk', 'CustomerId'], errors='ignore')
    y = df['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:,1]
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "f1": f1_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, proba)
            }
            for k,v in metrics.items():
                mlflow.log_metric(k,v)
            mlflow.log_param("model_name", name)
            joblib.dump(model, f"models/{name}.pkl")
            mlflow.log_artifact(f"models/{name}.pkl")

if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed.csv")
    train_models(df)
