import os
import hashlib
import hmac
import secrets

from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from database import get_connection, initialize_database


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

ENROLLMENT_KEY = os.getenv("CLOUDWISE_ENROLLMENT_KEY")

app = FastAPI(title="CloudWise AI Secure Backend")

initialize_database()


# ==========================================
# REQUEST MODELS
# ==========================================

class DeviceEnrollment(BaseModel):
    device_id: str = Field(min_length=1, max_length=100)
    device_name: str = Field(min_length=1, max_length=100)
    operating_system: str = Field(default="Unknown", max_length=100)


class DeviceMetrics(BaseModel):
    device_id: str = Field(min_length=1, max_length=100)
    device_name: str = Field(min_length=1, max_length=100)
    operating_system: str = Field(default="Unknown", max_length=100)

    cpu_usage: float = Field(ge=0, le=100)
    ram_usage: float = Field(ge=0, le=100)
    disk_usage: float = Field(ge=0, le=100)

    active_jobs: int = Field(ge=0)
    idle_time_minutes: int = Field(ge=0)


# ==========================================
# SECURITY HELPERS
# ==========================================

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def verify_enrollment_key(provided_key: str):

    if not ENROLLMENT_KEY:
        raise HTTPException(
            status_code=503,
            detail="Enrollment security is not configured"
        )

    if not hmac.compare_digest(provided_key, ENROLLMENT_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid enrollment key"
        )


def require_enrollment_key(
    x_enrollment_key: str = Header(default=None)
):
    if not x_enrollment_key:
        raise HTTPException(
            status_code=401,
            detail="Enrollment key required"
        )

    verify_enrollment_key(x_enrollment_key)


def authenticate_device(
    device_id: str,
    authorization: str
):

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Device authentication required"
        )

    provided_token = authorization[7:].strip()

    if not provided_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid device token"
        )

    with get_connection() as connection:

        device = connection.execute("""
            SELECT token_hash
            FROM devices
            WHERE device_id = ?
        """, (device_id,)).fetchone()

    if not device or not device["token_hash"]:
        raise HTTPException(
            status_code=401,
            detail="Device is not enrolled"
        )

    provided_hash = hash_token(provided_token)

    if not hmac.compare_digest(
        provided_hash,
        device["token_hash"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid device token"
        )


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "CloudWise AI Secure Backend is running",
        "database": "Connected",
        "security": "Device authentication enabled"
    }


# ==========================================
# DEVICE ENROLLMENT
# ==========================================

@app.post("/enroll")
def enroll_device(
    device: DeviceEnrollment,
    x_enrollment_key: str = Header(default=None)
):

    if not x_enrollment_key:
        raise HTTPException(
            status_code=401,
            detail="Enrollment key required"
        )

    verify_enrollment_key(x_enrollment_key)

    # Generate a unique secret token for this device.
    device_token = secrets.token_urlsafe(32)

    token_hash = hash_token(device_token)

    registered_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:

        connection.execute("""
            INSERT INTO devices (
                device_id,
                device_name,
                operating_system,
                registered_at,
                last_seen,
                token_hash
            )
            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT(device_id)
            DO UPDATE SET
                device_name = excluded.device_name,
                operating_system = excluded.operating_system,
                registered_at = excluded.registered_at,
                token_hash = excluded.token_hash
        """, (
            device.device_id,
            device.device_name,
            device.operating_system,
            registered_at,
            registered_at,
            token_hash
        ))

    # The plaintext token is returned only during enrollment.
    return {
        "message": "Device enrolled successfully",
        "device_id": device.device_id,
        "device_name": device.device_name,
        "device_token": device_token
    }


# ==========================================
# RECEIVE DEVICE METRICS
# ==========================================

@app.post("/metrics")
def receive_metrics(
    data: DeviceMetrics,
    authorization: str = Header(default=None)
):

    authenticate_device(data.device_id, authorization)

    recorded_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:

        connection.execute("""
            UPDATE devices
            SET
                device_name = ?,
                operating_system = ?,
                last_seen = ?
            WHERE device_id = ?
        """, (
            data.device_name,
            data.operating_system,
            recorded_at,
            data.device_id
        ))

        connection.execute("""
            INSERT INTO metrics (
                device_id,
                cpu_usage,
                ram_usage,
                disk_usage,
                active_jobs,
                idle_time_minutes,
                recorded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            data.device_id,
            data.cpu_usage,
            data.ram_usage,
            data.disk_usage,
            data.active_jobs,
            data.idle_time_minutes,
            recorded_at
        ))

    return {
        "message": "Authenticated metrics saved successfully",
        "device_id": data.device_id
    }


# ==========================================
# GET CONNECTED DEVICES
# ==========================================

@app.get("/devices")
def get_devices(
    _: None = Depends(require_enrollment_key)
):

    with get_connection() as connection:

        devices = connection.execute("""
            SELECT
                d.device_id,
                d.device_name,
                d.operating_system,
                d.registered_at,
                d.last_seen,
                m.cpu_usage,
                m.ram_usage,
                m.disk_usage,
                m.active_jobs,
                m.idle_time_minutes
            FROM devices d
            LEFT JOIN metrics m
                ON m.id = (
                    SELECT id
                    FROM metrics
                    WHERE device_id = d.device_id
                    ORDER BY id DESC
                    LIMIT 1
                )
            ORDER BY d.last_seen DESC
        """).fetchall()

        return [dict(device) for device in devices]
    

# ==========================================
# GET DEVICE HISTORY
# ==========================================

@app.get("/history/{device_id}")
def get_device_history(
    device_id: str,
    limit: int = 100,
    _: None = Depends(require_enrollment_key)
):

    if not 1 <= limit <= 1000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 1000"
        )

    with get_connection() as connection:

        history = connection.execute("""
            SELECT *
            FROM metrics
            WHERE device_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (device_id, limit)).fetchall()

        return [dict(record) for record in history]
