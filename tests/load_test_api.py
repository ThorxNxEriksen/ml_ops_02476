from locust import HttpUser, task, between
from PIL import Image
import io

def dummy_image():
    """Create a simple test image"""
    img = Image.new('L', (224, 224), color=128)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.read()

class QuickDrawUser(HttpUser):
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)
    
    def on_start(self):
        """Create test image once when user starts"""
        self.test_image = dummy_image()

    @task(1)
    def test_root(self):
        """Test the root endpoint"""
        self.client.get("/")

    @task(2)
    def test_predict(self):
        """Test the predict endpoint"""
        files = {'file': ('test.png', self.test_image, 'image/png')}
        self.client.post("/predict", files=files)