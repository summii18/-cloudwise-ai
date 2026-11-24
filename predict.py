import pandas as pd
import joblib

# Load our saved model
model = joblib.load("cloudwise_model.pkl")

print("\n===================================")
print("       CLOUDWISE AI")
print(" Cloud Resource Recommendation")
print("===================================")

# Get resource information from the user
cpu = float(input("Enter CPU usage (%): "))
ram = float(input("Enter RAM usage (%): "))
jobs = int(input("Enter number of active jobs: "))
idle = float(input("Enter idle time (minutes): "))

# Validate the input values
if not 0 <= cpu <= 100:
    raise ValueError("CPU usage must be between 0 and 100.")

if not 0 <= ram <= 100:
    raise ValueError("RAM usage must be between 0 and 100.")

if jobs < 0 or idle < 0:
    raise ValueError("Active jobs and idle time cannot be negative.")

# Prepare input in the same feature order used during training
input_data = pd.DataFrame(
    [[cpu, ram, jobs, idle]],
    columns=[
        "cpu_usage_percent",
        "ram_usage_percent",
        "active_jobs",
        "idle_time_minutes",
    ],
)

# Predict a recommendation
prediction = model.predict(input_data)[0]

print("\n-----------------------------------")
print("CloudWise AI Recommendation:", prediction)
print("-----------------------------------")

print("\nNote: This is an experimental prediction")
print("based on the training dataset.")
print("Do not use it to shut down real servers.")


