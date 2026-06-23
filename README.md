# OrnithoAI - Bird Species Classifier

OrnithoAI is a modular, containerized machine learning application designed for bird species classification. The stack utilizes **FastAPI** for high-performance REST APIs, **Streamlit** for a clean, user-friendly frontend interface, **PyTorch** and **timm** for model serving, and **Docker Compose** for end-to-end orchestration.

---

## 🌟 Key Features

1. **Pretrained ImageNet Baseline**: The training service instantiates a pretrained ImageNet backbone (default: `resnet50`) via `torchvision.models`, fine-tunes it in two phases on the **CUB-200-2011** dataset (200 bird species), and saves the weights to allow immediate prediction serving.
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
│   │   └── train.py           # Two-phase fine-tuning: frozen head → full backbone (ResNet + AdamW + AMP)
│   ├── .dockerignore          # Exclude caches, environment, weights
│   ├── Dockerfile             # Python 3.13 Dockerfile (Exposes FastAPI on port 8000)
│   └── requirements.txt       # fastapi, uvicorn, torch, torchvision, timm, pillow
├── frontend/
│   ├── app.py                 # Streamlit UI with image preview & top-5 prediction bars
│   ├── .dockerignore          # Exclude caches
│   ├── Dockerfile             # Python 3.13 Dockerfile (Exposes Streamlit on port 8501)
│   └── requirements.txt       # streamlit, requests, pillow
├── docker-compose.yml         # Defines backend, frontend, trainer, healthchecks, and volumes
├── pyproject.toml             # Ruff (linter/formatter) and Pylint configuration
├── run.sh                     # Local run script for Linux/macOS
├── run_dla_windowsiakow.ps1   # Local run script for Windows (PowerShell)
├── .env.example               # Template environment configuration — copy to .env before running
└── README.md                  # This file
```

---

## 📦 Dataset

This project uses the **CUB-200-2011** (Caltech-UCSD Birds) dataset — 11,788 images across 200 bird species.

1. Download it from the [official source](https://www.vision.caltech.edu/datasets/cub_200_2011/) or via:
   ```bash
   wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
   tar -xzf CUB_200_2011.tgz -C data/raw/
   ```
2. The expected directory structure is `data/raw/CUB_200_2011/images/<class_folder>/*.jpg`.

> **Note:** Neither the dataset nor the model weights are included in the repository (size). Run the trainer service first to generate `models/bird_classifier.pt` before starting the backend.

---

## 🚀 Getting Started

### 🐳 Option A — Docker (recommended)

Make sure you have [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.

Copy the environment template:
```bash
cp .env.example .env
```

Build and start all services with a single command:
```bash
docker compose up --build -d
```

This triggers the following sequence:
1. **Model Generation**: The `trainer` service runs `backend/train/train.py`, downloads a pretrained `resnet50` backbone, saves weights to the shared `./models` directory, and exits (unless `SKIP_TRAIN` is set to `true`).
2. **API Startup**: The `backend` container waits for the trainer weights, loads them onto the detected hardware (CUDA or CPU), and marks itself healthy.
3. **Frontend Boot**: The `frontend` Streamlit container starts once the backend is healthy.

**Stop and retain model weights:**
```bash
docker compose down
```

**Wipe model weights:**
Simply delete the `./models/` folder from the host directory.
```bash
rm -rf models/
```

---

### 🐍 Option B — Local (Python 3.13 + venv)

**Prerequisites:** Python 3.13, `pip`.

Create and activate a virtual environment, then install all dependencies:
```bash
python3.13 -m venv .venv_suml
source .venv_suml/bin/activate          # Linux/macOS
# .venv_suml\Scripts\Activate.ps1      # Windows PowerShell

pip install -r backend/requirements.txt -r frontend/requirements.txt
```

Copy the environment template:
```bash
cp .env.example .env
```

**Linux/macOS** — start everything with one script:
```bash
./run.sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File run_dla_windowsiakow.ps1
```

Both scripts run the trainer, wait for the backend to become healthy, then start the frontend.

---

## 🌐 Service URLs

| Service | URL |
|---------|-----|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Swagger UI | http://localhost:8000/docs |
| Backend Health Check | http://localhost:8000/health |

---

## 🔍 Verification & Troubleshooting

**Check container statuses (Docker):**
```bash
docker compose ps
```
Both `bird-classifier-backend` and `bird-classifier-frontend` should show `healthy`.

**Monitor logs (Docker):**
```bash
docker logs bird-classifier-trainer
docker logs -f bird-classifier-backend
```