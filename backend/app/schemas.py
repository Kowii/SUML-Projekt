
from pydantic import BaseModel


class PredictionItem(BaseModel):
    species: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]
