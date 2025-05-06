# promote_latest.py
import mlflow
from mlflow.tracking import MlflowClient

def promote_latest_model(model_name: str):
    client = MlflowClient()

    # 获取所有已注册版本
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'.")

    # 找到最新版本（根据 version 编号最大）
    latest_version = max(versions, key=lambda v: int(v.version))

    # Promote 为 Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Promoted model '{model_name}' version {latest_version.version} to Production.")

if __name__ == "__main__":
    promote_latest_model("fraud-xgb")
