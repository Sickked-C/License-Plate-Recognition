"""
Tests for License Plate Recognition API
Run: pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import io

# ─────────────────────────────────────────
# Mock models để test không cần load YOLO/VietOCR thật
# ─────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_models():
    """Mock YOLO và VietOCR để test nhanh, không cần GPU."""
    with patch('app.main._state', {
        "yolo": MagicMock(),
        "ocr": MagicMock()
    }):
        yield


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image():
    """Tạo ảnh giả để test upload."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[30:70, 50:150] = (255, 255, 255)  # vẽ hình chữ nhật trắng
    _, buffer = cv2.imencode('.jpg', img)
    return io.BytesIO(buffer.tobytes())


# ─────────────────────────────────────────
# Tests
# ─────────────────────────────────────────
class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_message(self, client):
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "model" in data
        assert "docs" in data


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"


class TestPredictEndpoint:
    def test_predict_valid_jpg(self, client, sample_image):
        """Upload ảnh JPG hợp lệ."""
        # Mock YOLO trả về không có biển số
        from app.main import _state
        mock_result = MagicMock()
        mock_result.boxes = []
        _state["yolo"].return_value = [mock_result]

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        assert response.status_code == 200

    def test_predict_returns_correct_fields(self, client, sample_image):
        """Response phải có đủ các field cần thiết."""
        from app.main import _state
        mock_result = MagicMock()
        mock_result.boxes = []
        _state["yolo"].return_value = [mock_result]

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        data = response.json()
        assert "filename" in data
        assert "plates_found" in data
        assert "plates" in data
        assert "processing_time_s" in data

    def test_predict_invalid_format(self, client):
        """Upload file không phải ảnh phải trả về 400."""
        fake_pdf = io.BytesIO(b"fake pdf content")
        response = client.post(
            "/predict",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")}
        )
        assert response.status_code == 400

    def test_predict_no_plates_found(self, client, sample_image):
        """Ảnh không có biển số → plates_found = 0."""
        from app.main import _state
        mock_result = MagicMock()
        mock_result.boxes = []
        _state["yolo"].return_value = [mock_result]

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image, "image/jpeg")}
        )
        data = response.json()
        assert data["plates_found"] == 0
        assert data["plates"] == []

    def test_predict_png_supported(self, client, sample_image):
        """PNG cũng phải được chấp nhận."""
        from app.main import _state
        mock_result = MagicMock()
        mock_result.boxes = []
        _state["yolo"].return_value = [mock_result]

        response = client.post(
            "/predict",
            files={"file": ("test.png", sample_image, "image/png")}
        )
        assert response.status_code == 200


class TestCorrectPlateText:
    """Test hàm correct_plate_text riêng."""

    def test_correct_dash(self):
        from app.main import correct_plate_text
        assert correct_plate_text("51G.12345") == "51G-12345"

    def test_correct_number_in_suffix(self):
        from app.main import correct_plate_text
        result = correct_plate_text("51G-1234O")
        assert "0" in result  # O → 0

    def test_preserve_prefix_letters(self):
        from app.main import correct_plate_text
        result = correct_plate_text("89A-07379")
        assert result.startswith("89A")  # A trong prefix không bị đổi

    def test_handles_space(self):
        from app.main import correct_plate_text
        result = correct_plate_text("51G 12345")
        assert " " not in result


class TestSplitTwoLinePlate:
    """Test hàm split_two_line_plate."""

    def test_square_plate_gets_split(self):
        from app.main import split_two_line_plate
        # Ảnh vuông → biển 2 dòng → nên được ghép ngang
        square_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = split_two_line_plate(square_img)
        h, w = result.shape[:2]
        assert w > h  # kết quả phải rộng hơn cao

    def test_wide_plate_unchanged(self):
        from app.main import split_two_line_plate
        # Ảnh ngang → biển 1 dòng → giữ nguyên
        wide_img = np.zeros((50, 200, 3), dtype=np.uint8)
        result = split_two_line_plate(wide_img)
        assert result.shape == wide_img.shape