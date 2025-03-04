import mlflow
import os

# Set the tracking URI (change if using a remote server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set up the experiment
mlflow.set_experiment("MLflow Overview")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("optimizer", "adam")

    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.35)

    # Create a sample artifact
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/info.txt", "w") as f:
        f.write("This is a sample artifact in MLflow.")

    # Log artifact
    mlflow.log_artifact("artifacts/info.txt")
