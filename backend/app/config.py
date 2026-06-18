import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_DIR: str = "/models"
    MODEL_FILE: str = "bird_classifier.pt"
    CLASSES_FILE: str = "classes.json"
    MODEL_NAME: str = "resnet50"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @property
    def model_path(self) -> str:
        return os.path.join(self.MODEL_DIR, self.MODEL_FILE)

    @property
    def classes_path(self) -> str:
        return os.path.join(self.MODEL_DIR, self.CLASSES_FILE)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
