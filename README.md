# 🚗 Vietnamese License Plate Recognition

![CI](https://github.com/Sickked-C/License-Plate-Recognition/actions/workflows/ci.yml/badge.svg)

A production-ready pipeline that detects and reads Vietnamese motorcycle license plates using **YOLOv8** for detection and **VietOCR** for character recognition — served via **FastAPI**, containerized with **Docker**, and tested with **CI/CD**.

🌐 **[Live Demo](https://sickked-c.github.io/License-Plate-Recognition/)** · 📖 **[API Docs](https://github.com/Sickked-C/License-Plate-Recognition)**

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **99.4%** |
| **mAP@0.5:0.95** | **91.3%** |
| Precision | 98.3% |
| Recall | 99.0% |

### Per-class Detection

| Class | Description | mAP@0.5 |
|-------|-------------|---------|
| BSD | Biển số dọc (1 dòng) | 99.5% |
| BSV | Biển số vuông (2 dòng) | 99.3% |

---

## 🔍 How it works

```
Input Image
      ↓
YOLOv8n — detect & crop license plate region
      ↓
split_two_line_plate() — handle 2-line plates
      ↓
preprocess_plate() — CLAHE + denoise + sharpen
      ↓
VietOCR — extract character string
      ↓
correct_plate_text() — fix common OCR errors
      ↓
Output: { "text": "51G-123.45", "confidence": 0.994 }
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8n (Ultralytics 8.3) |
| OCR | VietOCR (Transformer + VGG) |
| API Framework | FastAPI |
| Container | Docker |
| CI/CD | GitHub Actions |
| Dataset | Kaggle — Nguyễn Duy Diệu (~1,500 images) |
| Training | Google Colab (Tesla T4 GPU) |

---

## 🚀 Quick Start

### Run with Docker

```bash
# Pull and run
docker build -t lpr-api .
docker run -p 8000:8000 lpr-api
```

Visit **http://localhost:8000/docs**

### Run locally

```bash
# 1. Clone repo
git clone https://github.com/Sickked-C/License-Plate-Recognition.git
cd License-Plate-Recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
uvicorn app.main:app --reload
```

---

## 📡 API Usage

### Predict license plate

```bash
POST /predict

curl -X POST http://localhost:8000/predict \
  -F "file=@motorcycle.jpg"
```

**Response:**
```json
{
  "filename": "motorcycle.jpg",
  "plates_found": 1,
  "plates": [
    {
      "text": "51G-123.45",
      "text_raw": "51G-123.45",
      "confidence": 0.994,
      "bbox": [135, 105, 210, 169]
    }
  ],
  "processing_time_s": 0.167
}
```

---

## ⚙️ CI/CD Pipeline

```
git push
    ↓
GitHub Actions:
  1. pytest tests/test_api.py (15 tests)
  2. docker build -t lpr-api .
```

Every push to `main` automatically runs tests and builds the Docker image.

---

## 📁 Project Structure

```
.
├── app/
│   └── main.py              # FastAPI + YOLOv8 + VietOCR pipeline
├── tests/
│   └── test_api.py          # 15 unit & integration tests
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI/CD
├── index.html               # Web UI demo
├── best.pt                  # Trained YOLOv8 model
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧪 Tests

```bash
pytest tests/test_api.py -v
```

```
15 passed in 11.63s ✅
```

Test coverage includes: endpoint responses, input validation, plate text correction, two-line plate splitting.

---

## 💡 Key Takeaways

- **YOLOv8n achieves near-perfect detection** (99.4% mAP) with a lightweight nano model
- **Two-line plates** require preprocessing — split and merge before OCR
- **Post-processing rules** improve OCR accuracy for common character confusions
- **Docker bakes model weights** into image — no download at runtime

---

## ⚠️ Known Limitations

- Digits `5/6` and `0/8` may be confused in shadowed or low-resolution images
- Strong oblique angles reduce OCR accuracy
- Improvement: fine-tune VietOCR specifically on license plate dataset

---

## 📄 License

MIT License