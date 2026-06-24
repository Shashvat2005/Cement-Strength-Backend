from fastapi import FastAPI

from Predition_Input import PredictionInput, BatchPredictionInput
from database import save_prediction
from predict import predict_strength, predict_strength_batch
from fastapi.middleware.cors import CORSMiddleware
import time

from supabase_client import supabase

app = FastAPI(title = "Cement Strength Backend")


app.add_middleware(

    CORSMiddleware,
    allow_origins=["*"],  # replace with frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        prediction = predict_strength(data)
        # print(prediction)
        return {
            "success": True,
            "prediction": prediction
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
@app.post("/batch_predict")
async def predict_batch(data: BatchPredictionInput):
    return predict_strength_batch(data)
@app.put("/actual-strength/{prediction_id}")
async def update_actual_strength(
    prediction_id: int,
    actual_strength: float
):

    response = supabase.table(
        "predictions"
    ).update({
        "actual_strength_28d": actual_strength,
        "prediction_verified": True
    }).eq(
        "id",
        prediction_id
    ).execute()

    return response