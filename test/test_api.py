from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.post("/score_batch", json={"lines": ["invalid log"] * 32})
    assert response.status_code == 422
