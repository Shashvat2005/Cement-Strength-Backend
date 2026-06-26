from pydantic import BaseModel

class PredictionInput(BaseModel):
    date: str
    plant: str
    blaine: float
    residue90: float
    residue45: float
    loi: float
    so3: float
    c3s: float
    c2s: float
    twoDays: float
    sevenDays: float

class BatchPredictionInput(BaseModel):
    raw_data: str
    plant: str