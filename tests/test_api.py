from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check_get():
    """Test GET /health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data

def test_list_models_get():
    """Test GET /models"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "models" in data
    assert data["total"] >= 1

def test_get_model_detail_get():
    """Test GET /models/{model_id}"""
    response = client.get("/models/linear_regression")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "linear_regression"
    assert data["is_active"] is True

def test_create_model_post():
    """Test POST /models"""
    payload = {
        "model_id": "test_decision_tree",
        "name": "Test Decision Tree",
        "version": "1.0.0",
        "description": "Test model for API unit test",
        "framework": "scikit-learn",
        "is_active": True,
        "hyperparameters": {"max_depth": 5},
        "tags": ["test", "tree"]
    }
    response = client.post("/models", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["model_id"] == "test_decision_tree"
    assert data["name"] == "Test Decision Tree"

def test_update_model_put():
    """Test PUT /models/{model_id}"""
    payload = {
        "name": "Updated Decision Tree",
        "version": "2.0.0",
        "description": "Fully updated decision tree",
        "framework": "scikit-learn",
        "is_active": True,
        "hyperparameters": {"max_depth": 10},
        "tags": ["tree", "updated"]
    }
    response = client.put("/models/test_decision_tree", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "2.0.0"
    assert data["name"] == "Updated Decision Tree"

def test_patch_model_patch():
    """Test PATCH /models/{model_id}"""
    payload = {
        "is_active": False,
        "description": "Patched description"
    }
    response = client.patch("/models/test_decision_tree", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_active"] is False
    assert data["description"] == "Patched description"
    # Unchanged fields should remain
    assert data["version"] == "2.0.0"

def test_delete_model_delete():
    """Test DELETE /models/{model_id}"""
    response = client.delete("/models/test_decision_tree")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    # Confirm model no longer exists
    get_res = client.get("/models/test_decision_tree")
    assert get_res.status_code == 404

def test_predict_post():
    """Test POST /predict"""
    payload = {
        "features": {
            "area_sqft": 1500,
            "bedrooms": 3,
            "bathrooms": 2,
            "age": 5,
            "parking": 1,
            "location_score": 8.5
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "latency_ms" in data
    assert "prediction_id" in data

def test_predict_by_model_post():
    """Test POST /predict/{model_id}"""
    payload = {
        "features": {
            "area_sqft": 2000,
            "bedrooms": 4,
            "bathrooms": 3,
            "age": 2,
            "parking": 2,
            "location_score": 9.0
        }
    }
    response = client.post("/predict/random_forest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "random_forest"
    assert "prediction" in data

def test_predict_batch_post():
    """Test POST /predict/{model_id}/batch"""
    payload = {
        "items": [
            {"area_sqft": 1200, "bedrooms": 2, "bathrooms": 1, "age": 10, "parking": 1, "location_score": 6.0},
            {"area_sqft": 3000, "bedrooms": 5, "bathrooms": 4, "age": 1, "parking": 3, "location_score": 9.5}
        ]
    }
    response = client.post("/predict/random_forest/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 2
    assert len(data["predictions"]) == 2

def test_history_get_and_delete():
    """Test GET /predict/history and DELETE /predict/history"""
    # Check history contains records
    history_res = client.get("/predict/history")
    assert history_res.status_code == 200
    records = history_res.json()
    assert isinstance(records, list)
    assert len(records) > 0

    # Clear history
    del_res = client.delete("/predict/history")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Verify history is now empty
    history_empty_res = client.get("/predict/history")
    assert history_empty_res.status_code == 200
    assert len(history_empty_res.json()) == 0
