"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel


class PredictionItem(BaseModel):
    """A single species prediction with its confidence score."""

    species: str
    confidence: float


class PredictionResponse(BaseModel):
    """Top-K species predictions returned by the /predict endpoint."""

    predictions: list[PredictionItem]
