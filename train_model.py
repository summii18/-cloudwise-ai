import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/Cost_Optimization.csv")

# Experiment 1: Use all seven features
all_features = [
    "cpu_usage_percent",
    "ram_usage_percent",
    "active_jobs",
    "idle_time_minutes",
    "power_consumption_watts",
    "utilization_score",
    "monthly_cost_usd",
]

# Experiment 2: Use only four basic resource features
basic_features = [
    "cpu_usage_percent",
    "ram_usage_percent",
    "active_jobs",
    "idle_time_minutes",
]

# Target to predict
y = df["recommendation"]

# Split once so both experiments use the same records
X_train_all, X_test_all, y_train, y_test = train_test_split(
    df[all_features],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train model using all features
model_all = DecisionTreeClassifier(
    max_depth=6,
    random_state=42,
)

model_all.fit(X_train_all, y_train)

predictions_all = model_all.predict(X_test_all)

accuracy_all = accuracy_score(y_test, predictions_all)

# Train model using only basic features
X_train_basic = X_train_all[basic_features]
X_test_basic = X_test_all[basic_features]

model_basic = DecisionTreeClassifier(
    max_depth=6,
    random_state=42,
)

model_basic.fit(X_train_basic, y_train)

predictions_basic = model_basic.predict(X_test_basic)

accuracy_basic = accuracy_score(y_test, predictions_basic)

# Display results
print("CloudWise AI - Feature Comparison")
print("--------------------------------")

print(f"Accuracy using all 7 features: {accuracy_all:.2%}")
print(f"Accuracy using 4 basic features: {accuracy_basic:.2%}")

print("\nExperiment completed.")
