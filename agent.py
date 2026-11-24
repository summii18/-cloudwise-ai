import platform
import socket
import time
import json
from pathlib import Path

import psutil
import requests


# ==========================================
# LOAD DEVICE CONFIGURATION
# ==========================================

CONFIG_PATH = Path(__file__).resolve().parent / "device_config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        "device_config.json not found. Enroll this device first."
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

DEVICE_ID = config["device_id"]
DEVICE_TOKEN = config["device_token"]
API_URL = config["api_url"].rstrip("/") + "/metrics"

if not DEVICE_TOKEN:
    raise ValueError("Device token is missing.")


# ==========================================
# IDLE TIME TRACKING
# ==========================================

idle_seconds = 0
last_sample_time = time.monotonic()


# ==========================================
# COLLECT DEVICE METRICS
# ==========================================

def collect_metrics():

    global idle_seconds, last_sample_time

    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent

    disk = psutil.disk_usage("C:\\").percent

    active_jobs = len(psutil.pids())

    current_time = time.monotonic()
    elapsed = current_time - last_sample_time
    last_sample_time = current_time

    if cpu < 10:
        idle_seconds += elapsed
    else:
        idle_seconds = 0

    idle_minutes = int(idle_seconds // 60)

    return {
        "device_id": DEVICE_ID,
        "device_name": socket.gethostname(),
        "operating_system": platform.system(),
        "cpu_usage": cpu,
        "ram_usage": ram,
        "disk_usage": disk,
        "active_jobs": active_jobs,
        "idle_time_minutes": idle_minutes
    }


# ==========================================
# SEND AUTHENTICATED METRICS
# ==========================================

def start_monitoring():

    headers = {
        "Authorization": f"Bearer {DEVICE_TOKEN}"
    }

    print("CloudWise AI Secure Monitoring Agent Started")
    print("Device:", socket.gethostname())
    print("Authentication: Enabled")
    print("Press CTRL+C to stop.\n")

    while True:

        metrics = collect_metrics()

        print("CPU:", metrics["cpu_usage"], "%")
        print("RAM:", metrics["ram_usage"], "%")
        print("Disk:", metrics["disk_usage"], "%")
        print("Active processes:", metrics["active_jobs"])
        print("Idle time:", metrics["idle_time_minutes"], "minutes")

        try:

            response = requests.post(
                API_URL,
                json=metrics,
                headers=headers,
                timeout=5
            )

            if response.ok:
                print("Backend:", response.json()["message"])
            else:
                print("Backend error:", response.status_code)
                print("Details:", response.text)

        except requests.exceptions.RequestException as error:
            print("Connection error:", error)

        print("-" * 50)

        time.sleep(5)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    try:
        start_monitoring()

    except KeyboardInterrupt:
        print("\nCloudWise AI Monitoring Agent stopped.")
