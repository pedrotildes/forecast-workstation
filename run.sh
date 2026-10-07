#!/usr/bin/env bash
# Arranca o Forecast Workstation em http://localhost:8600
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3.12 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
cd backend
exec ../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8600}" "$@"
