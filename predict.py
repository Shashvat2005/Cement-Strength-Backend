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

    # print(df.values.tolist())

    # df["Date"] = pd.to_datetime(
    #     df["Date"],
    #     format="%d.%m.%Y"
    # )
    df["Date"] = pd.to_datetime(df["Date"])
    # print(df.values.tolist())

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter

    df.drop(columns=["Date"], inplace=True)

    prediction = float(model.predict(df)[0])
    # print(prediction)

    prediction_time_ms = round(
        (time.perf_counter() - start_time) * 1000
    )

    if save:
        save_prediction(
            data=data,
            prediction=prediction,
            prediction_time_ms=prediction_time_ms
        )

    return prediction

def predict_strength_batch(data: BatchPredictionInput):

    results = []

    rows = data.raw_data.splitlines()

    for row in rows:

        values = row.split()

        if len(values) != 11:
            continue

        sample = SimpleNamespace(
            date=values[0],
            plant=int(values[1]),
            blaine=float(values[2]),
            residue90=float(values[3]),
            residue45=float(values[4]),
            loi=float(values[5]),
            so3=float(values[6]),
            c3s=float(values[7]),
            c2s=float(values[8]),
            twoDays=float(values[9]),
            sevenDays=float(values[10])
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