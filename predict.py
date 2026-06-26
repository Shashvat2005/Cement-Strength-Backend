import datetime

import pandas as pd
import os
from huggingface_hub import hf_hub_download, login
import joblib
from types import SimpleNamespace

from Predition_Input import BatchPredictionInput
import time

from database import save_prediction

HF_TOKEN = os.getenv("HF_TOKEN")

model_path = hf_hub_download(
    repo_id="starintern/cement-strength-model",
    filename="stacking_regressor.pkl",
    token = HF_TOKEN
)

model = joblib.load(model_path)


def predict_strength(data, save=True):
    # print(data)

    start_time = time.perf_counter()

    df = pd.DataFrame([{
        "Date": data.date,
        "Plant": data.plant,
        "Blaine": data.blaine,
        "Residue 90": data.residue90,
        "Residue 45": data.residue45,
        "L0I": data.loi,
        "SO3": data.so3,
        "C3S": data.c3s,
        "C2S": data.c2s,
        "2 Days": data.twoDays,
        "7 Days": data.sevenDays
    }])


    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter

    # print(df.head())

    df.drop(columns=["Date"], inplace=True)
    plant_map = {
        "ACI": 0,
        "ACF": 1,
        "SSCI": 2,
        "SCI": 3,
    }

    df["Plant"] = df["Plant"].map(plant_map).fillna(4).astype(int)

    # print(df.head())

    prediction = float(model.predict(df)[0])

    # print("Predict", prediction)

    prediction_time_ms = round(
        (time.perf_counter() - start_time) * 1000
    )

    if save:
        try:
            data.plant = plant_map[data.plant]
        except Exception as e:
            print(e)
        save_prediction(
            data=data,
            prediction=prediction,
            prediction_time_ms=prediction_time_ms
        )

    return prediction

def predict_strength_batch(data: BatchPredictionInput):

    results = []
        # print(data)

    rows = data.raw_data.splitlines()
    plant = data.plant

    for row in rows:

        values = row.split()

        if len(values) != 10:
            continue

        sample = SimpleNamespace(
            date=values[0],
            plant=plant,
            blaine=float(values[1]),
            residue90=float(values[2]),
            residue45=float(values[3]),
            loi=float(values[4]),
            so3=float(values[5]),
            c3s=float(values[6]),
            c2s=float(values[7]),
            twoDays=float(values[8]),
            sevenDays=float(values[9])
        )

        prediction = predict_strength(sample)

        results.append({
            "date": sample.date,
            "plant": sample.plant,
            "blaine": sample.blaine,
            "residue90": sample.residue90,
            "residue45": sample.residue45,
            "loi": sample.loi,
            "so3": sample.so3,
            "c3s": sample.c3s,
            "c2s": sample.c2s,
            "twoDays": sample.twoDays,
            "sevenDays": sample.sevenDays,
            "prediction": round(float(prediction), 2)
        })

    return {
        "results": results
    }