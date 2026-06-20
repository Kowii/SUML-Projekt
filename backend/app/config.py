"""Application configuration loaded from environment variables or .env file."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Katalog projektu: backend/app/config.py -> backend/app -> backend -> project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Pydantic settings model — values are read from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MODEL_DIR: str = str(_PROJECT_ROOT / "models")
    MODEL_FILE: str = "bird_classifier.pt"
    CLASSES_FILE: str = "classes.json"
    MODEL_NAME: str = "resnet50"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @property
    def model_path(self) -> str:
        """Return the absolute path to the PyTorch weights file."""
        return os.path.join(self.MODEL_DIR, self.MODEL_FILE)

    @property
    def classes_path(self) -> str:
        """Return the absolute path to the JSON class-mapping file."""
        return os.path.join(self.MODEL_DIR, self.CLASSES_FILE)
