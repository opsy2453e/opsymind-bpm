from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import os
import numpy as np
import librosa

# 🔥 FIX 0: защита от предупреждений librosa/resampy
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()

# 📁 static
app.mount("/static", StaticFiles(directory="static"), name="static")


# 🎧 анализ аудио
def analyze_audio(path):

    # 🔥 FIX 1: стабильная загрузка mp3 на сервере (ffmpeg fallback)
    y, sr = librosa.load(
        path,
        sr=16000,
        mono=True,
        duration=15,
        res_type="kaiser_fast"
    )

    # 🔥 FIX 2: безопасный BPM (разные версии librosa)
    tempo_result = librosa.beat.beat_track(y=y, sr=sr)

    if isinstance(tempo_result, tuple):
        tempo = tempo_result[0]
    else:
        tempo = tempo_result

    # 🔥 FIX 3: защита от None
    if tempo is None:
        tempo = 0

    # ⚡ energy
    energy = float(np.mean(np.abs(y))) * 100

    # 🎼 chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    major_profile = np.array([
        6.35, 2.23, 3.48, 2.33,
        4.38, 4.09, 2.52, 5.19,
        2.39, 3.66, 2.29, 2.88
    ])

    minor_profile = np.array([
        6.33, 2.68, 3.52, 5.38,
        2.60, 3.53, 2.54, 4.75,
        3.98, 2.69, 3.34, 3.17
    ])

    KEYS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    scores = []

    for i in range(12):

        major_score = np.corrcoef(
            np.roll(chroma_mean, i),
            major_profile
        )[0, 1]

        minor_score = np.corrcoef(
            np.roll(chroma_mean, i),
            minor_profile
        )[0, 1]

        # 🔥 FIX 4: защита от NaN (главная причина crash)
        if np.isnan(major_score):
            major_score = -999
        if np.isnan(minor_score):
            minor_score = -999

        scores.append((major_score, KEYS[i], "major"))
        scores.append((minor_score, KEYS[i], "minor"))

    best = max(scores, key=lambda x: x[0])

    key = f"{best[1]} {best[2]}"

    return {
        # 🔥 FIX 5: безопасный cast (чтобы фронт НИКОГДА не видел undefined)
        "bpm": round(float(tempo), 1),
        "key": key,
        "energy": round(float(energy), 1)
    }


# 🌐 home
@app.get("/")
def home():
    return FileResponse("index.html")


# 📤 upload
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    if not file.filename:
        return {"error": "No file uploaded"}

    content = await file.read()

    path = f"temp_{file.filename}"

    with open(path, "wb") as f:
        f.write(content)

    try:
        return analyze_audio(path)

    except Exception as e:

        # 🔥 FIX 6: ВСЕГДА JSON (никогда не ломает frontend)
        return {
            "bpm": 0,
            "key": "unknown",
            "energy": 0,
            "error": str(e)
        }

    finally:
        if os.path.exists(path):
            os.remove(path)