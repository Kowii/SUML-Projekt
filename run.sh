#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv_suml"
MODELS_DIR="$SCRIPT_DIR/models"

PYTHON="$VENV/bin/python"
UVICORN="$VENV/bin/uvicorn"
STREAMLIT="$VENV/bin/streamlit"

cleanup() {
    echo ""
    echo "Stopping services..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    echo "Stopped."
}

trap cleanup INT TERM

mkdir -p "$MODELS_DIR"

# --- TRAINER ---
echo ">>> Running trainer..."
MODEL_DIR="$MODELS_DIR" "$PYTHON" "$SCRIPT_DIR/backend/train/train.py"
echo ">>> Trainer done."

# --- BACKEND ---
echo ">>> Starting backend..."
MODEL_DIR="$MODELS_DIR" PYTHONPATH="$SCRIPT_DIR/backend" \
    "$UVICORN" app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo ">>> Waiting for backend to become healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo ">>> Backend healthy."
        break
    fi
    sleep 1
done

# --- FRONTEND ---
echo ">>> Starting frontend..."
BACKEND_URL="http://localhost:8000" \
    "$STREAMLIT" run "$SCRIPT_DIR/frontend/app.py" --server.headless=true &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "  Frontend : http://localhost:8501"
echo "  API docs : http://localhost:8000/docs"
echo "  Health   : http://localhost:8000/health"
echo "========================================"
echo "Press Ctrl+C to stop."

wait "$BACKEND_PID" "$FRONTEND_PID"