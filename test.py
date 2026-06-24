import os
import psutil
import joblib

process = psutil.Process(os.getpid())

print("Before load:", process.memory_info().rss / 1024**2, "MB")

model = joblib.load("model/stacking_regressor.pkl")

print("After load:", process.memory_info().rss / 1024**2, "MB")