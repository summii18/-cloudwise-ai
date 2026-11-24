
import random

FEATURES = [
    "cpu_usage_percent",
    "ram_usage_percent",
    "active_jobs",
    "idle_time_minutes",
]

def generate_simulated_servers(dataset, count=10):
    records = dataset[FEATURES].sample(
        n=count,
        replace=True
    ).reset_index(drop=True)

    servers = []

    for i, record in records.iterrows():
        servers.append({
            "server_id": f"EC2-SIM-{i + 1:02d}",
            "metrics": {
                "cpu_usage_percent": float(
                    record["cpu_usage_percent"]
                ),
                "ram_usage_percent": float(
                    record["ram_usage_percent"]
                ),
                "active_jobs": int(
                    record["active_jobs"]
                ),
                "idle_time_minutes": float(
                    record["idle_time_minutes"]
                ),
            }
        })

    return servers

def evolve_simulated_servers(servers, dataset):

    cpu_min = dataset["cpu_usage_percent"].min()
    cpu_max = dataset["cpu_usage_percent"].max()

    ram_min = dataset["ram_usage_percent"].min()
    ram_max = dataset["ram_usage_percent"].max()

    jobs_min = int(dataset["active_jobs"].min())
    jobs_max = int(dataset["active_jobs"].max())

    idle_min = dataset["idle_time_minutes"].min()
    idle_max = dataset["idle_time_minutes"].max()

    for server in servers:
        metrics = server["metrics"]

        old_jobs = metrics["active_jobs"]

        new_jobs = max(
            jobs_min,
            min(jobs_max, old_jobs + random.choice([-1, 0, 1]))
        )

        metrics["cpu_usage_percent"] = round(
            max(
                cpu_min,
                min(
                    cpu_max,
                    metrics["cpu_usage_percent"]
                    + random.uniform(-5, 5)
                    + (new_jobs - old_jobs) * 2
                )
            ),
            2
        )

        metrics["ram_usage_percent"] = round(
            max(
                ram_min,
                min(
                    ram_max,
                    metrics["ram_usage_percent"]
                    + random.uniform(-2, 2)
                    + (new_jobs - old_jobs)
                )
            ),
            2
        )

        metrics["active_jobs"] = new_jobs

        if new_jobs == 0:
            metrics["idle_time_minutes"] = min(
                idle_max,
                metrics["idle_time_minutes"] + random.randint(1, 5)
            )
        else:
            metrics["idle_time_minutes"] = max(
                idle_min,
                metrics["idle_time_minutes"] - random.randint(1, 5)
            )

    return servers