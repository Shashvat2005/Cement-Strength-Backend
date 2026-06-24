from datetime import datetime

from supabase_client import supabase
import pandas as pd

MODEL_VERSION = "stack_1555"

def normalize_date(date_str):
    return pd.to_datetime(
        date_str
    ).strftime("%Y-%m-%d")

def save_prediction(
        data,
        prediction,
        prediction_time_ms
):

    supabase.table("predictions").upsert(
        {
            "date": normalize_date(data.date),
            "plant": data.plant,

            "blaine": data.blaine,
            "residue90": data.residue90,
            "residue45": data.residue45,

            "loi": data.loi,
            "so3": data.so3,

            "c3s": data.c3s,
            "c2s": data.c2s,

            "two_days": data.twoDays,
            "seven_days": data.sevenDays,

            "predicted_strength": prediction,

            "model_version": MODEL_VERSION,
            "prediction_time_ms": prediction_time_ms
        },
        on_conflict="date,plant"
    ).execute()
