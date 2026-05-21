import os
import json
import logging
import asyncio
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self, model_path: str, classes_path: str, model_name: str):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Any = None
        self.classes: List[str] = []
        self.transform: Any = None

    def load_model(self) -> None:
        logger.info(f"Using device: {self.device}")
        
        # Load classes mapping
        logger.info(f"Loading class mapping from {self.classes_path}")
        with open(self.classes_path, "r") as f:
            self.classes = json.load(f)
        
        logger.info(f"Classes loaded ({len(self.classes)} total): {self.classes}")

        # Instantiate model architecture
        logger.info(f"Instantiating model structure for '{self.model_name}'...")
        self.model = timm.create_model(self.model_name, pretrained=False, num_classes=len(self.classes))
        
        # Load state dictionary
        logger.info(f"Loading model weights from {self.model_path}")
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        logger.info("Model loaded successfully and ready for inference!")

    async def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Ensure that the model file exists and the service is initialized.")
        
        # Offload the heavy compute (Pillow transform, tensor copying, forward pass, softmax) to a background thread
        # This keeps the FastAPI asyncio event loop unblocked and free to serve other clients!
        return await asyncio.to_thread(self._run_inference, image)

    def _run_inference(self, image: Image.Image) -> List[Dict[str, Any]]:
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Transform and batch image (adding batch dimension)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run synchronous PyTorch forward pass without tracking gradients
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Determine the top-5 classes or total classes if less than 5
        top_k = min(5, len(self.classes))
        confidences, indices = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_k):
            results.append({
                "species": self.classes[indices[i].item()],
                "confidence": float(confidences[i].item())
            })
        
        return results
