import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Load the dataset
df = pd.read_csv("data/Cost_Optimization.csv")

# Select the same features used during training
features = [
    "cpu_usage_percent",
    "ram_usage_percent",
    "active_jobs",
    "idle_time_minutes",
]

X = df[features]
y = df["recommendation"]

# Recreate the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Load the saved model
model = joblib.load("cloudwise_model.pkl")

# Generate predictions for the test data
predictions = model.predict(X_test)

# Display evaluation results
print("CLOUDWISE AI - MODEL EVALUATION")
print("=" * 40)

print(f"Test samples: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))
print("\nClass order:", list(model.classes_))
