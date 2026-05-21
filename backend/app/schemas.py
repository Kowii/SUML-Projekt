from pydantic import BaseModel
from typing import List

class PredictionItem(BaseModel):
    species: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]
