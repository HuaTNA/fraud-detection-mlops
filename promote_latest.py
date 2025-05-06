# promote_latest.py
import mlflow
from mlflow.tracking import MlflowClient

def promote_latest_model(model_name: str):
    client = MlflowClient()

    
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'.")

    latest_version = max(versions, key=lambda v: int(v.version))


    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Promoted model '{model_name}' version {latest_version.version} to Production.")

if __name__ == "__main__":
    promote_latest_model("fraud-xgb")
