import sqlite3
from pathlib import Path

# DATABASE CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cloudwise.db"


# DATABASE CONNECTION

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# INITIALIZE DATABASE

def initialize_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                device_name TEXT NOT NULL,
                operating_system TEXT,
                registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_seen TEXT
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                cpu_usage REAL NOT NULL,
                ram_usage REAL NOT NULL,
                disk_usage REAL NOT NULL,
                active_jobs INTEGER NOT NULL,
                idle_time_minutes INTEGER NOT NULL,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (device_id)
                    REFERENCES devices(device_id)
            )
        """)

        # Safely migrate existing devices table.
        columns = connection.execute(
            "PRAGMA table_info(devices)"
        ).fetchall()

        column_names = [column["name"] for column in columns]

        if "token_hash" not in column_names:
            connection.execute("""
                ALTER TABLE devices
                ADD COLUMN token_hash TEXT
            """)

        # Index for faster device history queries.
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_device_id
            ON metrics(device_id)
        """)


# VERIFY DATABASE

def verify_database():

    with get_connection() as connection:

        tables = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
        """).fetchall()

        print("\nDatabase tables:")

        for table in tables:
            print("-", table["name"])

        columns = connection.execute(
            "PRAGMA table_info(devices)"
        ).fetchall()

        print("\nDevice table columns:")

        for column in columns:
            print("-", column["name"])

        record_count = connection.execute(
            "SELECT COUNT(*) FROM metrics"
        ).fetchone()[0]

        print("\nHistorical metric records:", record_count)


# MAIN

if __name__ == "__main__":

    initialize_database()

    print("\nCloudWise AI database migration completed!")

    verify_database()