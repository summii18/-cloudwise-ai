import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/Cost_Optimization.csv")

# Use four basic resource features
features = [
    "cpu_usage_percent",
    "ram_usage_percent",
    "active_jobs",
    "idle_time_minutes",
]

X = df[features]
y = df["recommendation"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train the model
model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42,
)

model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "cloudwise_model.pkl")

print("CloudWise AI model saved successfully!")
print("Model file: cloudwise_model.pkl")
print("Input features:", features)
