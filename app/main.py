import re
import time

import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# ─────────────────────────────────────────
# State
# ─────────────────────────────────────────
_state = {"yolo": None, "ocr": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Loading YOLOv8 model...")
    _state["yolo"] = YOLO("best.pt")

    print("[STARTUP] Loading VietOCR model...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    _state["ocr"] = Predictor(config)

    print("[STARTUP] Ready!")
    yield

# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────
app = FastAPI(title="License Plate Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────
def split_two_line_plate(plate_img):
    """Tách biển số 2 dòng thành 1 dòng ghép lại."""
    h, w = plate_img.shape[:2]
    if h / w > 0.5:
        top = plate_img[0:h//2, :]
        bottom = plate_img[h//2:, :]
        return np.hstack([top, bottom])
    return plate_img


def preprocess_plate(plate_img):
    """Tăng chất lượng ảnh biển số trước OCR."""
    h, w = plate_img.shape[:2]
    plate_img = cv2.resize(plate_img, (w * 3, h * 3),
                           interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # CLAHE — xử lý bóng
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def correct_plate_text(text: str) -> str:
    text = re.sub(r'\s', '', text.strip().upper())

    # Thêm: chuẩn hoá dấu phân cách
    text = re.sub(r'[:\.](?=\d{3})', '-', text)  # : hoặc . trước 3 số → -

    match = re.match(r'^(\d{2}[A-Z]{1,2}\d?)[.\-]([0-9A-Z.]+)$', text)

    if match:
        prefix = match.group(1)
        suffix = match.group(2)

        # Sửa prefix — 2 số đầu chỉ được là số
        LETTER_TO_NUM = {'O': '0', 'I': '1', 'S': '5',
                         'B': '8', 'G': '6', 'Z': '2'}
        prefix_digits = prefix[:2]
        prefix_letters = prefix[2:]

        # Fix 2 số đầu nếu bị nhầm chữ
        prefix_digits_fixed = ''.join(
            LETTER_TO_NUM.get(c, c) for c in prefix_digits
        )

        # Sửa suffix
        NUMBER_FIX = {'O': '0', 'I': '1', 'S': '5',
                      'B': '8', 'A': '4', 'T': '7', 'G': '6'}
        suffix_fixed = ''.join(
            NUMBER_FIX.get(c, c) if not c.isdigit() and c != '.' else c
            for c in suffix
        )

        return f"{prefix_digits_fixed}{prefix_letters}-{suffix_fixed}"

    return text

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "License Plate Recognition API",
        "model": "YOLOv8 + VietOCR",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolo_loaded": _state["yolo"] is not None,
        "ocr_loaded": _state["ocr"] is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, .png supported")

    start = time.time()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image")

    try:
        results = _state["yolo"](img, verbose=False)
        plates = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            plate_crop = split_two_line_plate(plate_crop)
            plate_crop = preprocess_plate(plate_crop)
            plate_pil = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))

            text_raw = _state["ocr"].predict(plate_pil)
            text = correct_plate_text(text_raw)

            plates.append({
                "text": text,
                "text_raw": text_raw,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

        return {
            "filename": file.filename,
            "plates_found": len(plates),
            "plates": plates,
            "processing_time_s": round(time.time() - start, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))