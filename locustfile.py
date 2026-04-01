import os
from io import BytesIO

from locust import HttpUser, between, task
from PIL import Image

DEFAULT_HOST = os.environ.get("LOCUST_HOST", "http://127.0.0.1:8000")


def _jpeg_bytes():
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(200, 100, 120)).save(buf, format="JPEG")
    return buf.getvalue()


class ModelUser(HttpUser):
    host = DEFAULT_HOST
    wait_time = between(1, 3)

    @task
    def predict_image(self):
        path = os.path.join("data", "test", "sample_image.jpg")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
        else:
            data = _jpeg_bytes()
        self.client.post(
            "/predict",
            files={"file": ("sample.jpg", data, "image/jpeg")},
        )
