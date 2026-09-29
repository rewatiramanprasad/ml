# FastAPI Model Serving API Documentation

A production-ready, reusable FastAPI template for machine learning model serving featuring full RESTful routes: `GET`, `POST`, `PUT`, `PATCH`, and `DELETE`.

---

## 🚀 Quick Start

### 1. Run the FastAPI Application
```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Landing Dashboard**: [http://localhost:8000](http://localhost:8000)

---

## 📌 Available Endpoints & HTTP Methods

| HTTP Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Service health status and metadata |
| `GET` | `/models` | List all registered ML models |
| `GET` | `/models/{model_id}` | Get model configuration and hyperparameters |
| `POST` | `/models` | Register a new ML model in the registry |
| `PUT` | `/models/{model_id}` | Full update / replace model configuration |
| `PATCH` | `/models/{model_id}` | Partial update (e.g. toggle active state, update tags) |
| `DELETE` | `/models/{model_id}` | Delete / unregister model |
| `POST` | `/predict` | Run inference with default model |
| `POST` | `/predict/{model_id}` | Run inference with a specific model |
| `POST` | `/predict/{model_id}/batch` | Run batch inference on multiple inputs |
| `GET` | `/predict/history` | Retrieve prediction history logs |
| `DELETE` | `/predict/history` | Clear prediction history |

---

## 🧪 Example API Requests (cURL)

### 1. `GET /health` — Check Health Status
```bash
curl -X GET "http://localhost:8000/health"
```

### 2. `GET /models` — List All Models
```bash
curl -X GET "http://localhost:8000/models"
```

### 3. `POST /models` — Register a New Model
```bash
curl -X POST "http://localhost:8000/models" \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "xgboost_v1",
       "name": "XGBoost Regressor",
       "version": "1.0.0",
       "description": "Gradient boosted tree model",
       "framework": "xgboost",
       "is_active": true,
       "hyperparameters": {
         "n_estimators": 200,
         "learning_rate": 0.05
       },
       "tags": ["boosting", "production"]
     }'
```

### 4. `PUT /models/{model_id}` — Replace Full Model Config
```bash
curl -X PUT "http://localhost:8000/models/xgboost_v1" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "XGBoost Regressor v2",
       "version": "2.0.0",
       "description": "Retrained with tuned hyperparameters",
       "framework": "xgboost",
       "is_active": true,
       "hyperparameters": {
         "n_estimators": 300,
         "learning_rate": 0.03
       },
       "tags": ["boosting", "v2"]
     }'
```

### 5. `PATCH /models/{model_id}` — Partial Update
```bash
curl -X PATCH "http://localhost:8000/models/xgboost_v1" \
     -H "Content-Type: application/json" \
     -d '{
       "is_active": false,
       "description": "Temporarily disabled for maintenance"
     }'
```

### 6. `DELETE /models/{model_id}` — Delete / Unregister Model
```bash
curl -X DELETE "http://localhost:8000/models/xgboost_v1"
```

### 7. `POST /predict` — Predict With Default Model
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "area_sqft": 1500,
         "bedrooms": 3,
         "bathrooms": 2,
         "age": 5,
         "parking": 1,
         "location_score": 8.5
       }
     }'
```

### 8. `POST /predict/{model_id}` — Predict With Specific Model
```bash
curl -X POST "http://localhost:8000/predict/random_forest" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "area_sqft": 2000,
         "bedrooms": 4,
         "bathrooms": 3,
         "age": 2,
         "parking": 2,
         "location_score": 9.0
       }
     }'
```

### 9. `POST /predict/{model_id}/batch` — Batch Inference
```bash
curl -X POST "http://localhost:8000/predict/random_forest/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "items": [
         {"area_sqft": 1200, "bedrooms": 2, "bathrooms": 1, "age": 10, "parking": 1, "location_score": 6.0},
         {"area_sqft": 3000, "bedrooms": 5, "bathrooms": 4, "age": 1, "parking": 3, "location_score": 9.5}
       ]
     }'
```

### 10. `GET /predict/history` & `DELETE /predict/history`
```bash
# Get history
curl -X GET "http://localhost:8000/predict/history?limit=10"

# Clear history
curl -X DELETE "http://localhost:8000/predict/history"
```

---

## 🔁 How to Repeat for Any Model in the Future
1. Define your model features in [api/schemas.py](file:///Users/apple/Desktop/innovate/ml/api/schemas.py).
2. Register the model using `POST /models` or add it to `api/services.py`.
3. Call `POST /predict/{your_model_id}` to run inference!
