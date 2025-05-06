import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature


mlflow.set_tracking_uri("file:/app/mlruns")
mlflow.set_experiment("fraud") 

def train_and_log_model():
    
    df = pd.read_csv("data/creditcard.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    
    neg, pos = np.bincount(y)
    scale = neg / pos


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale
        )

        model.fit(X_train, y_train)


        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        recall = recall_score(y_test, preds)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("scale_pos_weight", scale)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", recall)

        print("Confusion matrix:\n", confusion_matrix(y_test, preds))

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[0:1]

        mlflow.xgboost.log_model(
            model,
            artifact_path="xgb_model",
            registered_model_name="fraud-xgb",
            signature=signature,
            input_example=input_example
        )

        print(f"Model logged with acc={acc:.4f}, recall={recall:.4f}")

if __name__ == "__main__":
    train_and_log_model()
