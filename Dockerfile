FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download VietOCR model lúc build — không cần download lúc run
RUN python -c "from vietocr.tool.config import Cfg; \
    from vietocr.tool.predictor import Predictor; \
    config = Cfg.load_config_from_name('vgg_transformer'); \
    config['device'] = 'cpu'; \
    Predictor(config)"

COPY app/ ./app/
COPY best.pt .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]