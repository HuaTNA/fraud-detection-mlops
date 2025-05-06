import mlflow.pyfunc

def load_model():
    model = mlflow.pyfunc.load_model("models:/fraud-xgb/Production")
    return model
