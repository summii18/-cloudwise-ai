import pandas as pd
import joblib

from simulation import generate_simulated_metrics

# Load existing model and dataset
model = joblib.load("cloudwise_model.pkl")
df = pd.read_csv("data/Cost_Optimization.csv")

features = [
    "cpu_usage_percent",
    "ram_usage_percent",
    "active_jobs",
    "idle_time_minutes"
]

print("\nCLOUDWISE AI - SIMULATION + ML TEST\n")

# Generate five simulations and predict recommendations
for i in range(5):

    metrics = generate_simulated_metrics(df)

    input_data = pd.DataFrame([metrics], columns=features)

    prediction = model.predict(input_data)[0]

    print(f"Simulation {i + 1}: {metrics}")
    print(f"AI Recommendation: {prediction}\n")