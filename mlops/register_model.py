import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")


mlflow.set_experiment("fraud-detection-exp")


client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("fraud-detection-exp")
runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)

if not runs:
    raise Exception("No MLflow runs found. Please train the model first.")

run_id = runs[0].info.run_id
print(f"Found latest run ID: {run_id}")


model_uri = f"runs:/{run_id}/xgb_model"
result = mlflow.register_model(model_uri=model_uri, name="fraud-xgb")

# Transition to Production
client.transition_model_version_stage(
    name="fraud-xgb",
    version=result.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Model registered and promoted to Production: fraud-xgb v{result.version}")
