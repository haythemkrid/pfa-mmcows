print('hello mmcows')
import mlflow
import os

# 1. Point directly to your local UI server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Create a dedicated test experiment
mlflow.set_experiment("Connection_Test")

print("Starting MLflow connection test...")

# 3. Start a run and log fake data
with mlflow.start_run(run_name="ping_test"):
    # Log a fake parameter
    mlflow.log_param("test_mode", "active")
    
    # Log a fake metric
    mlflow.log_metric("dummy_accuracy", 0.99)
    
    # Create a small text file and log it as an artifact
    artifact_name = "hello_mlflow.txt"
    with open(artifact_name, "w") as f:
        f.write("If you can read this in the UI, your artifacts are saving perfectly!")
    
    mlflow.log_artifact(artifact_name)

print("Test complete! Go check http://127.0.0.1:5000")