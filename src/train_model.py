import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Load data
df = pd.read_csv("../data/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test = X[:200000], X[200000:]
y_train, y_test = y[:200000], y[200000:]

mlflow.set_experiment("fraud-detection-exp")
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Train model
    model = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Log metrics
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)

    # Save and log model
    mlflow.xgboost.log_model(model, "xgb_model", registered_model_name="fraud-xgb")
    print("Model logged and registered.")
    
