import sys
from unittest.mock import MagicMock

# Mock pandas to avoid binary compatibility issues
sys.modules.setdefault("pandas", MagicMock())

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
resp = client.get("/admin/explain?prompt=test")
print("Status:", resp.status_code)
print("Body:", resp.text)
