"""Inference service — loads a torchvision ResNet model and runs non-blocking predictions."""

import asyncio
import json
import logging
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, tuple] = {
    "resnet18": (models.resnet18, 512),
    "resnet50": (models.resnet50, 2048),
}

_INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class InferenceService:
    """Thread-safe image classification service backed by a torchvision ResNet model."""

    def __init__(self, model_path: str, classes_path: str, model_name: str):
        """Store model file paths, architecture name, and detect the compute device."""
        self.model_path = model_path
        self.classes_path = classes_path
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.classes: list[str] = []

    def load_model(self) -> None:
        """Load class mapping, instantiate the backbone, replace head, and apply saved weights."""
        logger.info("Using device: %s", self.device)

        logger.info("Loading class mapping from %s", self.classes_path)
        with open(self.classes_path, encoding="utf-8") as f:
            self.classes = json.load(f)
        logger.info("Classes loaded (%d total): %s", len(self.classes), self.classes)

        if self.model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. Choose from: {list(_MODEL_REGISTRY)}"
            )

        model_fn, fc_in_features = _MODEL_REGISTRY[self.model_name]
        logger.info("Instantiating model architecture: %s", self.model_name)
        backbone = model_fn(weights=None)
        backbone.fc = nn.Linear(fc_in_features, len(self.classes))

        logger.info("Loading model weights from %s", self.model_path)
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        backbone.load_state_dict(state_dict)
        backbone.to(self.device)
        backbone.eval()

        self.model = backbone
        logger.info("Model loaded successfully and ready for inference!")

    async def predict(self, image: Image.Image) -> list[dict[str, Any]]:
        """Run inference asynchronously by offloading the forward pass to a background thread."""
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. "
                "Ensure the model file exists and the service is initialized."
            )
        return await asyncio.to_thread(self._run_inference, image)

    def _run_inference(self, image: Image.Image) -> list[dict[str, Any]]:
        """Execute the synchronous PyTorch forward pass and return top-K results."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = _INFERENCE_TRANSFORMS(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        top_k = min(5, len(self.classes))
        confidences, indices = torch.topk(probabilities, top_k)

        return [
            {
                "species": self.classes[indices[i].item()],
                "confidence": float(confidences[i].item()),
            }
            for i in range(top_k)
        ]
