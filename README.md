# OrnithoAI - Bird Species Classifier

OrnithoAI is a modular, containerized machine learning application designed for bird species classification. The stack utilizes **FastAPI** for high-performance REST APIs, **Streamlit** for a clean, user-friendly frontend interface, **PyTorch** and **timm** for model serving, and **Docker Compose** for end-to-end orchestration.

---

## 🌟 Key Features

1. **Pretrained ImageNet Baseline**: The training service instantiates a pretrained ImageNet backbone (default: `resnet50`) via `timm`, dynamically resets the classification head to the target classes (sparrow, robin, pigeon, eagle, seagull), and saves the weights to allow immediate prediction serving.
2. **Dynamic Non-Blocking Backend Startup**: The backend polls for the shared model files upon startup with standard Python `logging`. If they are not found within **5 minutes (300s)**, the startup sequence times out and aborts safely to prevent deadlocks.
3. **Non-Blocking Inference Loop**: PyTorch image preprocessing and forward propagation tasks are offloaded to background threads using `asyncio.to_thread`. This keeps the main FastAPI event loop fully free to handle concurrent requests.
4. **Auto CUDA Fallback**: The inference service automatically detects GPU compatibility (`cuda`), routing calculations to CUDA when available and falling back transparently to CPU.
5. **Decoupled Architecture**: Model configurations, directories, weights files, and API service URLs are fully externalized via standard `.env` configuration. You can switch models (e.g. from `resnet50` to `efficientnet_b0`) without altering a single line of API routing or frontend code.
6. **Docker Healthchecks**: Standard container-level healthchecks are integrated into `docker-compose.yml` to monitor services, ensuring the Streamlit application starts only after the FastAPI backend registers a healthy status.

---

## 📂 Project Directory Structure

```
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI Application Entrypoint (waits for model with 5-min timeout)
│   │   ├── config.py          # Pydantic Settings-based config loading
│   │   ├── schemas.py         # Pydantic schemas for API inputs/outputs
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py      # GET /health
│   │   │   └── predict.py     # POST /predict (Uses asyncio.to_thread for non-blocking prediction)
│   │   └── services/
│   │       ├── __init__.py
│   │       └── inference.py   # Inference service with CUDA/CPU detection & thread safety
│   ├── train/
│   │   ├── __init__.py
│   │   └── train.py           # Pretrained ImageNet dummy model generator
│   ├── .dockerignore          # Exclude caches, environment, weights
│   ├── Dockerfile             # Multi-stage-like Python 3.11 Dockerfile (Exposes FastAPI)
│   └── requirements.txt       # fastapi, uvicorn, torch, torchvision, timm, pillow
├── frontend/
│   ├── app.py                 # Clean, minimal Streamlit UI with image preview & progress bars
│   ├── .dockerignore          # Exclude caches
│   ├── Dockerfile             # Streamlit Dockerfile (Exposes port 8501)
│   └── requirements.txt       # streamlit, requests, pillow
├── docker-compose.yml         # Defines backend, frontend, trainer, healthchecks, and volumes
├── .env.example               # Template environment configuration
├── .env                       # Local environment configuration
└── README.md                  # Detailed startup and orchestration instructions
```

---

## 🚀 Getting Started

### 📋 Prerequisites
Make sure you have [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed on your machine.

### 🛠️ Building & Starting the Application
Initialize and boot all components with a single command:
```bash
docker compose up --build -d
```

This triggers the following sequence:
1. **Model Generation**: The `trainer` service compiles `backend/train/train.py`, downloads a pretrained `resnet50` backbone, resets the classification layer to the target class list, saves weights to the shared `models_volume`, and exits cleanly.
2. **API Startup**: The `backend` container boots up, waits for the trainer to output weights, successfully loads the weights onto the detected hardware device (CUDA or CPU), and marks itself healthy once the `/health` endpoint is alive.
3. **Frontend Boot**: The `frontend` Streamlit container boots up once the backend container registers a `healthy` status, exposing the UI.

### 🌐 Interfacing with the Services
- **Streamlit Frontend**: Open your browser and navigate to **`http://localhost:8501`**.
- **FastAPI Documentation (Swagger UI)**: Navigate to **`http://localhost:8000/docs`**.
- **Backend Health Check**: Access **`http://localhost:8000/health`**.

### 🛑 Stopping the Application
To stop the services and retain the generated model weights:
```bash
docker compose down
```

To stop services and completely wipe the persistent model volume and datasets:
```bash
docker compose down -v
```

---

## 🔍 Verification & Troubleshooting

### Viewing Service Statuses
Verify all containers are up and running cleanly with:
```bash
docker compose ps
```
The statuses for `bird-classifier-backend` and `bird-classifier-frontend` should show `healthy`.

### Monitoring Container Logs
Inspect logs of the trainer generating weights and backend loading them:
```bash
# Check trainer output
docker logs bird-classifier-trainer

# Follow backend loading and startup checks
docker logs -f bird-classifier-backend
```
