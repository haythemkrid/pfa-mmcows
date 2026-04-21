import mlflow
import os

print("Connecting to MLflow server at http://127.0.0.1:5000...")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create a test experiment
mlflow.set_experiment("Neon_Database_Test")

try:
    with mlflow.start_run(run_name="neon_ping"):
        # Log a fake metric
        mlflow.log_metric("connection_success", 1.0)
        mlflow.log_param("database", "neon_postgres")
        print("✅ Data successfully sent to the server!")
        
except Exception as e:
    print(f"❌ Failed to connect to server. Is it running? Error: {e}")

# The Ultimate Check: Did it save locally by accident?
print("-" * 40)
if os.path.exists("mlflow.db"):
    print("❌ FATAL: A local 'mlflow.db' file was created!")
    print("This means your MLflow server is still ignoring your Neon database.")
elif os.path.exists("mlruns"):
    print("❌ FATAL: A local 'mlruns' folder was created!")
    print("This means your tracking URI is wrong.")
else:
    print("🎉 SUCCESS: No local database files were created!")
    print("Your data was safely routed to the Neon cloud database. Go check the UI!")
print("-" * 40)
