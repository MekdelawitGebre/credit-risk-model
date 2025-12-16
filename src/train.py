import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import joblib
import os


def load_data(path="data/processed/processed.csv"):
    """Load processed dataset."""
    df = pd.read_csv(path)
    X = df.drop(columns=['is_high_risk', 'CustomerId'], errors='ignore')
    y = df['is_high_risk']
    return X, y


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba)
    }
    return metrics


def train_and_log_models(X, y):
    """Train models, tune hyperparameters, and log results to MLflow."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models and their hyperparameter grids
    models = {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=1000),
            "param_grid": {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {"n_estimators": [100, 200], "max_depth": [5, 10, 20]}
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 0.2]}
        }
    }

    mlflow.set_experiment("credit-risk-model")

    for name, cfg in models.items():
        estimator = cfg["estimator"]
        param_grid = cfg["param_grid"]

        # Use GridSearchCV for hyperparameter tuning
        grid = GridSearchCV(estimator, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test)

        with mlflow.start_run(run_name=f"{name}_GridSearch"):
            # Log hyperparameters and metrics
            mlflow.log_param("model_name", name)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("best_cv_score", grid.best_score_)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Save and log the model artifact
            artifact_path = f"models/{name}_best.pkl"
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_model, artifact_path)
            mlflow.log_artifact(artifact_path)
            mlflow.sklearn.log_model(best_model, artifact_path=name)

            print(f"✅ {name} trained and logged to MLflow with best params: {grid.best_params_}")

    print("🎯 All models trained, tuned, and logged successfully.")


if __name__ == "__main__":
    X, y = load_data()
    train_and_log_models(X, y)
