import os
import mlflow
from dotenv import load_dotenv
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from functools import lru_cache

@lru_cache(maxsize=1)
def load_latest_mlflow_model(env_path=None) -> PyFuncModel:
    """
    Loads and caches the latest version of a model from MLflow Model Registry.

    Args:
        env_path (str): Optional path to the .env file.

    Returns:
        mlflow.pyfunc.PyFuncModel: The loaded model.
    """
    load_dotenv(dotenv_path=env_path)

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
    latest_version = max(versions, key=lambda v: int(v.version))

    model_uri = f"models:/{model_name}/{latest_version.version}"
    model = mlflow.pyfunc.load_model(model_uri)

    return model
