from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os
import tempfile
import numpy as np
import librosa

# 🔥 ДОБАВЛЕНО (пункт 4)
import soundfile as sf

app = FastAPI()

# 🌍 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# 📦 max upload size (20 MB)
MAX_FILE_SIZE = 20 * 1024 * 1024


# 🎧 audio analyze
def analyze_audio(path):

    # ⚡ load first 15 sec
    y, sr = librosa.load(
        path,
        sr=16000,
        mono=True,
        duration=15,
        res_type="kaiser_fast"  # 🔥 ДОБАВЛЕНО (стабильнее на сервере)
    )

    # 🔥 BPM
    tempo = librosa.beat.beat_track(
        y=y,
        sr=sr
    )[0]

    # ⚡ energy
    energy = float(np.mean(np.abs(y))) * 100

    # 🎼 chroma
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    chroma_mean = np.mean(chroma, axis=1)

    # 🎼 major/minor profiles
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

    KEYS = [
        'C', 'C#', 'D', 'D#',
        'E', 'F', 'F#', 'G',
        'G#', 'A', 'A#', 'B'
    ]

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

        scores.append((
            major_score,
            KEYS[i],
            "major"
        ))

        scores.append((
            minor_score,
            KEYS[i],
            "minor"
        ))

    best = max(scores, key=lambda x: x[0])

    key = f"{best[1]} {best[2]}"

    return {
        "bpm": round(float(tempo), 1),
        "key": key,
        "energy": round(energy, 1)
    }


# 🌐 home
@app.get("/")
def home():
    return FileResponse("index.html")


# 📤 upload analyze
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    if not file.filename:
        return {"error": "No file uploaded"}

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        return {"error": "File too large (max 20MB)"}

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        path = tmp.name

    try:
        result = analyze_audio(path)
        return result

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(path):
            os.remove(path)