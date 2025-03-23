"""
This script promotes the latest MLflow model version in 'Staging' to 'Production'.

It assumes MLFLOW_TRACKING_URI is already set via environment variables or .env.
"""

import os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

def promote_model_to_production():
    load_dotenv()

    username = os.getenv("MLFLOW_USERNAME")
    password = os.getenv("MLFLOW_PASSWORD")
    host = os.getenv("MLFLOW_TRACKING_HOST")
    model_name = os.getenv("MODEL_NAME")

    if not all([username, password, host, model_name]):
        raise EnvironmentError("Missing one or more MLflow environment variables.")

    # Set tracking URI
    mlflow.set_tracking_uri(f"https://{username}:{password}@{host.replace('https://', '')}")

    # Get latest model version
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    staging_version = next((v for v in versions if v.current_stage == "Staging"), None)
    print(staging_version)

    if not staging_version:
        raise ValueError(f"❌ No model version in 'Staging' for {model_name}.")

    print(f"✅ Promoting version {staging_version.version} of model '{model_name}' to 'Production'...")

    # Promote the model
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"🎉 Model version {staging_version.version} successfully promoted to 'Production'.")

if __name__ == "__main__":
    promote_model_to_production()
