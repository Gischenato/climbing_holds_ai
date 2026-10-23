"""API FastAPI. Rodar: uvicorn backend.main:app --reload"""
import io
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.detectors import Detector, SamDetector

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SIDE = 2048  # lado maior máximo; fotos maiores são reduzidas antes do modelo

detector: Detector | None = None
# o modelo usa a GPU: uma detecção por vez
_gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = SamDetector()  # carregado uma única vez
    yield


app = FastAPI(title="Detector de agarras", lifespan=lifespan)


def read_image(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")  # respeita a orientação do celular
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Arquivo não é uma imagem válida")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.asarray(img)


@app.post("/detect")
def detect(file: UploadFile):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Imagem maior que 20 MB")
    image = read_image(data)
    with _gpu_lock:
        holds = detector.detect(image)
    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "holds": [{"id": i, **asdict(h)} for i, h in enumerate(holds)],
    }


FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
