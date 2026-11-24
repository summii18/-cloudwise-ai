#!/usr/bin/env bash
# Install CloudWise AI on Ubuntu 24.04 and keep both services on localhost.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this script on the Ubuntu server only."
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env in $ROOT"
  exit 1
fi

if [[ ! -f cloudwise.db ]]; then
  echo "Missing cloudwise.db in $ROOT"
  exit 1
fi

if [[ ! -f cloudwise_model.pkl || ! -f data/Cost_Optimization.csv ]]; then
  echo "Missing model or dataset"
  exit 1
fi

chmod 600 .env

if [[ ! -f cloudwise.pre-deploy.db ]]; then
  cp -a cloudwise.db cloudwise.pre-deploy.db
fi

count_rows() {
  python3 - <<'PY'
import sqlite3
connection = sqlite3.connect("cloudwise.db")
metrics = connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
devices = connection.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
print(f"metrics={metrics} devices={devices}")
PY
}

BEFORE="$(count_rows)"
echo "Database before setup: $BEFORE"

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip python3-dev build-essential

install_venv() {
  local python_bin="$1"
  rm -rf .venv
  "$python_bin" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
}

install_venv python3

.venv/bin/python deploy/check_model.py

AFTER_INSTALL="$(count_rows)"
if [[ "$BEFORE" != "$AFTER_INSTALL" ]]; then
  echo "Database counts changed during install: $BEFORE -> $AFTER_INSTALL"
  exit 1
fi

APP_USER="$(id -un)"
install_unit() {
  local name="$1"
  sed -e "s|__USER__|$APP_USER|g" -e "s|__APP_ROOT__|$ROOT|g" \
    "deploy/${name}.service" | sudo tee "/etc/systemd/system/${name}.service" >/dev/null
}


install_unit cloudwise-api
install_unit cloudwise-web

sudo systemctl daemon-reload
sudo systemctl enable cloudwise-api.service cloudwise-web.service
sudo systemctl restart cloudwise-api.service cloudwise-web.service

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS "http://127.0.0.1:8000/" >/dev/null; then
    break
  fi
  sleep 2
done

curl -fsS "http://127.0.0.1:8000/"
echo
DEVICES_CODE="$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/devices")"
if [[ "$DEVICES_CODE" != "401" ]]; then
  echo "Expected /devices without a key to return 401, got $DEVICES_CODE"
  exit 1
fi
echo "auth_ok devices_without_key=401"
AFTER_START="$(count_rows)"
if [[ "$BEFORE" != "$AFTER_START" ]]; then
  echo "Database counts changed after startup: $BEFORE -> $AFTER_START"
  exit 1
fi
echo "database_unchanged $AFTER_START"
echo "CloudWise AI is listening on 127.0.0.1 only."
