import joblib
from sklearn.tree import export_text

# Load the saved model
model = joblib.load("cloudwise_model.pkl")

# Define the input features in training order
features = [
    "cpu_usage_percent",
    "ram_usage_percent",
    "active_jobs",
    "idle_time_minutes",
]

# Display the learned decision rules
rules = export_text(model, feature_names=features)

print("CLOUDWISE AI - LEARNED DECISION RULES")
print("=" * 45)
print(rules)

