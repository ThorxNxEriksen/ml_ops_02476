import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from app.backend_api import app  # Import our fastapi app

# Create a test client inheriting from the FastAPI packages
client = TestClient(app)

@pytest.fixture
def dummy_image():
    """Create a mock 224x224 grayscale image for testing"""
    img = Image.new('L', (224, 224), color=128)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_read_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "QuickDraw API is running"}

def test_predict_endpoint_valid_image(dummy_image):
    """Test the predict endpoint with a valid image"""
    files = {"file": ("test_image.png", dummy_image, "image/png")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    assert "category" in response.json()
    assert "confidence" in response.json()
    assert isinstance(response.json()["category"], str)
    assert isinstance(response.json()["confidence"], float)
    assert 0 <= response.json()["confidence"] <= 1

def test_predict_endpoint_no_file():
    """Test the predict endpoint with no file"""
    response = client.post("/predict")
    assert response.status_code != 200

def test_categories_list():
    """Test that categories list is properly defined"""
    from app.backend_api import CATEGORIES
    
    assert isinstance(CATEGORIES, list)
    assert len(CATEGORIES) == 10 # exaclty 10 categories
    assert all(isinstance(c, str) for c in CATEGORIES)
    assert len(CATEGORIES) == len(set(CATEGORIES))  # Check for duplicates