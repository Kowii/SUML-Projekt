"""Prediction route — accepts an uploaded bird image and returns top-K species."""

import io
import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image

from app.schemas import PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse)
async def predict_bird(request: Request, image: UploadFile = File(...)):
    """Validate the uploaded image and return top-K species predictions."""
    # 1. Validate file extension/mime type
    if not image.content_type.startswith("image/"):
        logger.warning("Rejected non-image upload attempt with MIME type: %s", image.content_type)
        raise HTTPException(status_code=400, detail="File uploaded must be a valid image.")

    try:
        # 2. Read image data
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # 3. Fetch the service from application state
        inference_service = request.app.state.inference_service

        # 4. Perform non-blocking prediction
        predictions = await inference_service.predict(pil_image)
        return {"predictions": predictions}

    except Exception as e:
        logger.error("Prediction failed with error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during image classification: {str(e)}"
        ) from e
