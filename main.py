
import pandas as pd

file_path = "data/Cost_Optimization.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("\nDataset size (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nRecommendation counts:")
print(df["recommendation"].value_counts())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nBasic statistical summary:")
print(df.describe())

print("\nAverage resource usage for each recommendation:")

print(
    df.groupby("recommendation")[
        [
            "cpu_usage_percent",
            "ram_usage_percent",
            "active_jobs",
            "idle_time_minutes",
            "power_consumption_watts",
            "utilization_score",
            "monthly_cost_usd",
        ]
    ].mean().round(2)
)

print("\n--- Recommendation Rules Check ---")

print(
    df.groupby("recommendation")[
        ["cpu_usage_percent", "ram_usage_percent",
         "active_jobs", "idle_time_minutes"]
    ].mean()
)


print("\n--- Duplicate Records Check ---")

print("Total duplicate rows:", df.duplicated().sum())

print("\n--- Recommendation Labels by CPU Range ---")

df["cpu_range"] = pd.cut(
    df["cpu_usage_percent"],
    bins=[0, 20, 40, 60, 80, 100],
    include_lowest=True,
)

print(
    pd.crosstab(
        df["cpu_range"],
        df["recommendation"],
        normalize="index"
    ).round(3)
)


print("\n--- Recommendation Distribution Within CPU Ranges ---")

for lower, upper in [(0, 20), (20, 40), (40, 60)]:
    subset = df[
        (df["cpu_usage_percent"] > lower)
        & (df["cpu_usage_percent"] <= upper)
    ]

    print(f"\nCPU usage: {lower}%–{upper}%")
    print("Number of records:", len(subset))

    print(
        subset.groupby("recommendation")[
            ["ram_usage_percent", "active_jobs", "idle_time_minutes"]
        ].mean().round(2)
    )



