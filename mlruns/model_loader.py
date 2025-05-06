import mlflow

def load_model():
    mlflow.set_tracking_uri("http://mlflow:5000")
    model = mlflow.pyfunc.load_model("models:/fraud-xgb/Production")
    return model


