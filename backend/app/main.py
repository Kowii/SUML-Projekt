"""FastAPI application entry point — wires up lifespan, routes, and inference service."""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.routes import health, predict
from app.services.inference import InferenceService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("backend_api")

settings = Settings()


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Poll for model files on startup, load them into app state, then yield."""
    model_path = settings.model_path
    classes_path = settings.classes_path

    logger.info("FastAPI Backend Startup: Checking for required model files at:")
    logger.info("  - Weights path: %s", model_path)
    logger.info("  - Classes path: %s", classes_path)

    start_time = time.time()
    timeout = 300.0
    poll_interval = 5.0

    while not (os.path.exists(model_path) and os.path.exists(classes_path)):
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            logger.critical(
                "STARTUP TIMEOUT ERROR: Model files not found within %ss limit. Exiting.",
                timeout,
            )
            sys.exit(1)

        remaining = int(timeout - elapsed)
        logger.warning(
            "Model files not found yet. Polling again in %ss... %ss left until timeout.",
            poll_interval,
            remaining,
        )
        await asyncio.sleep(poll_interval)

    logger.info("All required model files detected. Initializing Inference Service...")

    try:
        service = InferenceService(
            model_path=model_path,
            classes_path=classes_path,
            model_name=settings.MODEL_NAME,
        )
        service.load_model()
        fastapi_app.state.inference_service = service
        logger.info("FastAPI initialization completed successfully.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.critical("FATAL ERROR: Failed to load model: %s", str(e), exc_info=True)
        sys.exit(1)

    yield


app = FastAPI(
    title="OrnithoAI - Bird Species Classifier Backend",
    description="A modular FastAPI backend serving a PyTorch-based bird classifier.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
