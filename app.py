from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os
import numpy as np
import librosa

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


def analyze_audio(path):
    y, sr = librosa.load(path, sr=16000, mono=True, duration=15)

    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    energy = float(np.mean(np.abs(y))) * 100

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    major_profile = np.array([
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
        2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ])

    minor_profile = np.array([
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
        2.54, 4.75, 3.98, 2.69, 3.34, 3.17
    ])

    KEYS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    scores = []

    for i in range(12):
        major_score = np.corrcoef(np.roll(chroma_mean, i), major_profile)[0,1]
        minor_score = np.corrcoef(np.roll(chroma_mean, i), minor_profile)[0,1]

        scores.append((major_score, KEYS[i], "major"))
        scores.append((minor_score, KEYS[i], "minor"))

    best = max(scores, key=lambda x: x[0])
    key = f"{best[1]} {best[2]}"

    return {
        "bpm": round(float(tempo), 1),
        "key": key,
        "energy": round(energy, 1)
    }


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    try:
        return analyze_audio(path)
    finally:
        os.remove(path)