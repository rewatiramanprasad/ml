# Regression ML Lab

A hands-on machine learning project for learning regression **from
implementation to deployment**.

The goal is to build the same ML workflow in two ways:

1.  **Jupyter / Google Colab notebooks** --- understand the mathematics,
    intuition, implementation, training, evaluation, and experiments.
2.  **FastAPI application** --- load the trained model and expose
    predictions through production-style REST APIs.

------------------------------------------------------------------------

## 1. Project Goal

Build a single regression-learning project containing multiple
regression algorithms.

For every algorithm, follow this lifecycle:

``` text
Understand
   ↓
Implement from scratch
   ↓
Implement with scikit-learn
   ↓
Train on dataset
   ↓
Evaluate
   ↓
Tune
   ↓
Save model
   ↓
Load model
   ↓
Expose prediction through FastAPI
```

The purpose is not only to learn how to call ML libraries, but to
understand what is happening underneath and then turn the trained model
into a usable backend service.

------------------------------------------------------------------------

# 2. Learning Roadmap

## Phase 1 --- ML Fundamentals

Before implementing individual algorithms:

-   Python for ML
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Dataset loading
-   Features and target
-   Train/test split
-   Validation data
-   Data preprocessing
-   Missing values
-   Categorical variables
-   Feature scaling
-   Feature engineering
-   Overfitting
-   Underfitting
-   Bias vs variance

------------------------------------------------------------------------

# 3. Regression Tree

``` text
Regression
│
├── 1. Linear Models
│   │
│   ├── Linear Regression
│   ├── Polynomial Regression
│   ├── Ridge Regression
│   ├── Lasso Regression
│   └── Elastic Net
│
├── 2. Tree-Based Models
│   │
│   ├── Decision Tree Regression
│   └── Random Forest Regression
│
├── 3. Boosting Models
│   │
│   ├── Gradient Boosting Regression
│   ├── XGBoost Regression
│   ├── LightGBM Regression
│   └── CatBoost Regression
│
├── 4. Distance / Kernel Based
│   │
│   ├── KNN Regression
│   └── Support Vector Regression
│
└── 5. Advanced / Optional
    │
    ├── Bayesian Regression
    ├── Random Forest + Feature Engineering
    ├── Stacking Regression
    └── Voting Regression
```

------------------------------------------------------------------------

# 4. Recommended Algorithm Order

Learn the algorithms in this order:

  \#   Algorithm                      Main Concept
  ---- ------------------------------ ---------------------------------------
  01   Linear Regression              Baseline + least squares
  02   Polynomial Regression          Non-linear relationships
  03   Ridge Regression               L2 regularization
  04   Lasso Regression               L1 regularization
  05   Elastic Net                    L1 + L2
  06   Decision Tree Regression       Recursive splitting
  07   Random Forest Regression       Bagging + multiple trees
  08   Gradient Boosting Regression   Sequential error correction
  09   XGBoost                        Optimized gradient boosting
  10   LightGBM                       Histogram-based boosting
  11   CatBoost                       Categorical-feature-friendly boosting
  12   KNN Regression                 Distance-based prediction
  13   SVR                            Margin + kernel methods
  14   Ensemble Regression            Combining models

Do not jump directly to XGBoost.

The earlier models provide the concepts needed to understand why
boosting and ensemble methods work.

------------------------------------------------------------------------

# 5. Project Folder Structure

``` text
regression-ml-lab/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   │   └── house_prices.csv
│   │
│   ├── processed/
│   │   └── cleaned_data.csv
│   │
│   └── sample/
│
├── notebooks/
│   │
│   ├── 00_ml_fundamentals.ipynb
│   ├── 01_eda_and_preprocessing.ipynb
│   │
│   ├── linear/
│   │   ├── 01_linear_regression_from_scratch.ipynb
│   │   └── 02_linear_regression_sklearn.ipynb
│   │
│   ├── polynomial/
│   │   └── polynomial_regression.ipynb
│   │
│   ├── regularization/
│   │   ├── ridge_regression.ipynb
│   │   ├── lasso_regression.ipynb
│   │   └── elastic_net.ipynb
│   │
│   ├── tree/
│   │   └── decision_tree_regression.ipynb
│   │
│   ├── ensemble/
│   │   ├── random_forest_regression.ipynb
│   │   └── ensemble_regression.ipynb
│   │
│   ├── boosting/
│   │   ├── gradient_boosting.ipynb
│   │   ├── xgboost.ipynb
│   │   ├── lightgbm.ipynb
│   │   └── catboost.ipynb
│   │
│   ├── distance/
│   │   ├── knn_regression.ipynb
│   │   └── svr.ipynb
│   │
│   └── 99_model_comparison.ipynb
│
├── src/
│   │
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linear_regression.py
│   │   ├── polynomial_regression.py
│   │   ├── ridge.py
│   │   ├── lasso.py
│   │   ├── elastic_net.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── gradient_boosting.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   ├── knn.py
│   │   └── svr.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── comparison.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py
│
├── models/
│   ├── linear_regression/
│   │   └── model.pkl
│   ├── ridge/
│   │   └── model.pkl
│   ├── random_forest/
│   │   └── model.pkl
│   └── xgboost/
│       └── model.json
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── routes/
│   │   ├── health.py
│   │   ├── prediction.py
│   │   └── models.py
│   │
│   ├── schemas/
│   │   └── prediction.py
│   │
│   ├── services/
│   │   ├── predictor.py
│   │   └── model_loader.py
│   │
│   └── dependencies.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
│
└── docs/
    ├── algorithms.md
    ├── experiments.md
    └── api.md
```

------------------------------------------------------------------------

# 6. Dataset

We will initially use a **house-price regression problem**.

Example features:

``` text
area_sqft
bedrooms
bathrooms
age
parking
location_score
```

Target:

``` text
price
```

Example:

``` text
area_sqft = 1500
bedrooms = 3
bathrooms = 2
age = 5
parking = 1
location_score = 8.5

             ↓

          MODEL

             ↓

Predicted price = ₹X
```

Later, we can replace the dataset with real-world datasets.

------------------------------------------------------------------------

# 7. Standard Notebook Structure

Every regression notebook should follow the same structure.

``` text
1. Problem Definition
2. Import Libraries
3. Load Dataset
4. Understand Dataset
5. Exploratory Data Analysis
6. Data Cleaning
7. Feature Engineering
8. Train/Test Split
9. Preprocessing
10. Algorithm Intuition
11. Mathematical Foundation
12. Implementation From Scratch
13. Train Model
14. Predictions
15. Evaluation
16. Visualization
17. Hyperparameter Tuning
18. Error Analysis
19. Save Model
20. Final Observations
```

This consistency will make the entire project easier to understand.

------------------------------------------------------------------------

# 8. Metrics

For every regression model, calculate at least:

### MAE

Mean Absolute Error.

``` text
MAE = average(|actual - predicted|)
```

Easy to interpret because it represents the average absolute prediction
error.

### MSE

Mean Squared Error.

``` text
MSE = average((actual - predicted)^2)
```

Large errors receive a larger penalty.

### RMSE

Root Mean Squared Error.

``` text
RMSE = sqrt(MSE)
```

Useful because the result is in the same unit as the target.

### R²

Coefficient of determination.

``` text
R²
```

Measures how much of the target variance is explained by the model.

------------------------------------------------------------------------

# 9. Model Comparison

At the end, create a common comparison table.

``` text
Model                 MAE       RMSE       R²
------------------------------------------------
Linear Regression     ...       ...        ...
Polynomial            ...       ...        ...
Ridge                 ...       ...        ...
Lasso                 ...       ...        ...
Elastic Net           ...       ...        ...
Decision Tree         ...       ...        ...
Random Forest         ...       ...        ...
Gradient Boosting     ...       ...        ...
XGBoost               ...       ...        ...
LightGBM              ...       ...        ...
CatBoost              ...       ...        ...
KNN                   ...       ...        ...
SVR                   ...       ...        ...
```

The goal is not simply to find the lowest number.

We will investigate:

-   Why did the model perform this way?
-   Is the model overfitting?
-   Which features matter?
-   How sensitive is it to preprocessing?
-   How much does tuning improve it?
-   How fast is training?
-   How fast is inference?
-   How large is the saved model?

------------------------------------------------------------------------

# 10. From Notebook to Production

The notebook is the **learning environment**.

FastAPI becomes the **inference environment**.

``` text
                 NOTEBOOK
                    │
                    │
              Train Model
                    │
                    ▼
             Evaluate Model
                    │
                    ▼
              Save Model
                    │
                    ▼
             models/model.pkl
                    │
                    │
                    ▼
                FastAPI
                    │
              Load Model
                    │
                    ▼
             POST /predict
                    │
                    ▼
              Prediction
```

------------------------------------------------------------------------

# 11. FastAPI Architecture

Example request:

``` http
POST /predict
Content-Type: application/json
```

Request:

``` json
{
  "area_sqft": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "age": 5,
  "parking": 1,
  "location_score": 8.5
}
```

Response:

``` json
{
  "prediction": 8250000,
  "model": "random_forest",
  "version": "1.0"
}
```

------------------------------------------------------------------------

# 12. FastAPI Learning Flow

For each trained model:

``` text
Notebook
   ↓
Train
   ↓
Validate
   ↓
Save model
   ↓
FastAPI model loader
   ↓
Pydantic request schema
   ↓
Prediction service
   ↓
REST endpoint
   ↓
JSON response
```

------------------------------------------------------------------------

# 13. FastAPI Endpoints

Initial API:

``` text
GET  /health
GET  /models
POST /predict
POST /predict/{model_name}
```

Example:

``` text
GET /health

Response:

{
  "status": "healthy"
}
```

------------------------------------------------------------------------

## Model-specific prediction

``` text
POST /predict/random-forest
POST /predict/xgboost
POST /predict/linear
```

This allows us to compare models through the API itself.

------------------------------------------------------------------------

# 14. Model Loader

The API should load the model once when the application starts.

Conceptually:

``` python
model = load_model("models/random_forest/model.pkl")
```

Then:

``` python
@app.post("/predict")
def predict(request: PredictionRequest):

    prediction = model.predict([
        [
            request.area_sqft,
            request.bedrooms,
            request.bathrooms,
            request.age,
            request.parking,
            request.location_score
        ]
    ])

    return {
        "prediction": prediction[0]
    }
```

We will later improve this architecture using a dedicated prediction
service.

------------------------------------------------------------------------

# 15. Important Production Concepts

After the basic API works, learn:

``` text
Model serialization
      ↓
Pickle / Joblib / JSON
      ↓
Model versioning
      ↓
Input validation
      ↓
Preprocessing pipeline
      ↓
Logging
      ↓
Exception handling
      ↓
Unit tests
      ↓
API tests
      ↓
Docker
      ↓
CI/CD
```

------------------------------------------------------------------------

# 16. Prevent Training/Inference Mismatch

A critical production rule:

**The preprocessing used during training must be exactly the
preprocessing used during inference.**

Bad:

``` text
Notebook
   ↓
Manual preprocessing
   ↓
Model
```

and then:

``` text
FastAPI
   ↓
Different manual preprocessing
   ↓
Model
```

Better:

``` text
Raw Input
   ↓
Preprocessing Pipeline
   ↓
Model
   ↓
Prediction
```

Save the complete pipeline whenever possible.

``` python
pipeline.fit(X_train, y_train)
```

Then:

``` python
joblib.dump(pipeline, "model.pkl")
```

FastAPI:

``` python
pipeline = joblib.load("model.pkl")

prediction = pipeline.predict(input_data)
```

------------------------------------------------------------------------

# 17. Experiments to Track

For every model, record:

``` text
Algorithm
Dataset
Features
Preprocessing
Hyperparameters
Training Time
Inference Time
MAE
MSE
RMSE
R²
Model Size
Notes
```

Example:

``` text
Algorithm: Random Forest
n_estimators: 200
max_depth: 10

MAE: ...
RMSE: ...
R²: ...

Observation:
Model performs well but appears to overfit when max_depth increases.
```

------------------------------------------------------------------------

# 18. From-Scratch Implementation Roadmap

We should not implement every advanced algorithm from scratch.

### Implement deeply from scratch

``` text
Linear Regression
Polynomial Regression
Ridge
Lasso
Decision Tree — simplified version
KNN Regression
```

### Understand + use established implementation

``` text
Random Forest
Gradient Boosting
XGBoost
LightGBM
CatBoost
SVR
```

The objective is to understand the mechanics without spending excessive
time rebuilding highly optimized production libraries.

------------------------------------------------------------------------

# 19. Linear Regression Deep Dive

First implementation:

``` text
Dataset
   ↓
X, y
   ↓
Train/Test Split
   ↓
Calculate coefficients
   ↓
Predictions
   ↓
Residuals
   ↓
MAE / RMSE / R²
```

Then compare:

``` text
Our Linear Regression
        VS
sklearn LinearRegression
```

We should verify that the predictions and coefficients are approximately
consistent.

------------------------------------------------------------------------

# 20. Regularization Roadmap

After Linear Regression:

``` text
Linear Regression
       │
       ├── Ridge
       │     └── L2 penalty
       │
       ├── Lasso
       │     └── L1 penalty
       │
       └── Elastic Net
             ├── L1
             └── L2
```

Focus on:

-   Why regularization is needed
-   Effect of large coefficients
-   Alpha / lambda
-   Feature selection
-   Overfitting
-   Bias/variance trade-off

------------------------------------------------------------------------

# 21. Tree-Based Roadmap

``` text
Decision Tree
      │
      ▼
Multiple Trees
      │
      ▼
Random Forest
      │
      ▼
Boosting
      │
      ├── Gradient Boosting
      ├── XGBoost
      ├── LightGBM
      └── CatBoost
```

Focus on:

-   Splitting
-   Nodes
-   Leaves
-   Depth
-   Impurity
-   Bagging
-   Random feature selection
-   Boosting
-   Learning rate
-   Number of estimators
-   Overfitting control

------------------------------------------------------------------------

# 22. Final Project Architecture

``` text
                    REGRESSION ML LAB
                           │
          ┌────────────────┴────────────────┐
          │                                 │
     EXPERIMENTATION                    PRODUCTION
          │                                 │
      notebooks/                           api/
          │                                 │
     Data + EDA                         FastAPI
          │                                 │
     Algorithms                       Model Loader
          │                                 │
     Evaluation                       Prediction
          │                                 │
     Experiments                      Validation
          │                                 │
          └──────────────┬──────────────────┘
                         │
                    models/
                         │
                  Trained Pipelines
```

------------------------------------------------------------------------

# 23. Final Learning Path

``` text
Phase 1
Python + NumPy + Pandas
        ↓
Phase 2
EDA + preprocessing
        ↓
Phase 3
Linear Regression
        ↓
Phase 4
Polynomial + Regularization
        ↓
Phase 5
Decision Trees
        ↓
Phase 6
Random Forest
        ↓
Phase 7
Gradient Boosting
        ↓
Phase 8
XGBoost / LightGBM / CatBoost
        ↓
Phase 9
KNN + SVR
        ↓
Phase 10
Model Comparison
        ↓
Phase 11
Model Serialization
        ↓
Phase 12
FastAPI
        ↓
Phase 13
Testing
        ↓
Phase 14
Docker
        ↓
Phase 15
CI/CD
        ↓
Phase 16
Production ML API
```

------------------------------------------------------------------------

# 24. Definition of Done

The project is considered complete when:

-   [ ] Multiple regression algorithms implemented
-   [ ] At least 5 algorithms understood from fundamentals
-   [ ] Several algorithms implemented from scratch
-   [ ] All models evaluated with common metrics
-   [ ] Hyperparameter tuning performed
-   [ ] Model comparison notebook created
-   [ ] Best-performing experiments documented
-   [ ] Model pipelines serialized
-   [ ] FastAPI prediction API created
-   [ ] Multiple models available through API
-   [ ] Pydantic request validation implemented
-   [ ] Unit tests added
-   [ ] API tests added
-   [ ] Dockerized
-   [ ] README contains setup and usage instructions
-   [ ] Example API requests documented
-   [ ] Training → model artifact → API workflow works end-to-end

------------------------------------------------------------------------

# 25. Starting Point

Do **not** build the entire project at once.

Start with:

``` text
01_eda_and_preprocessing.ipynb
        ↓
02_linear_regression_from_scratch.ipynb
        ↓
03_linear_regression_sklearn.ipynb
        ↓
models/linear_regression/
        ↓
FastAPI /predict
```

Once this complete loop works:

``` text
DATA
 ↓
TRAIN
 ↓
EVALUATE
 ↓
SAVE MODEL
 ↓
FASTAPI
 ↓
PREDICT
```

repeat the same architecture for the next regression algorithm.

That repeated loop is the core of the entire project.
