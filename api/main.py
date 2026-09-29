from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from api.config import settings
from api.routes import health_router, models_router, predict_router

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(health_router)
app.include_router(models_router)
app.include_router(predict_router)

@app.get("/", response_class=HTMLResponse, tags=["Root"])
def root():
    """
    Landing page detailing available API routes and Swagger links.
    """
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{settings.APP_TITLE}</title>
        <style>
            :root {{
                --bg-primary: #0f172a;
                --bg-card: #1e293b;
                --text-primary: #f8fafc;
                --text-secondary: #94a3b8;
                --accent-blue: #38bdf8;
                --accent-green: #4ade80;
                --accent-amber: #fbbf24;
                --accent-purple: #c084fc;
                --accent-rose: #fb7185;
                --border-color: #334155;
            }}
            * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
            body {{ background-color: var(--bg-primary); color: var(--text-primary); padding: 40px 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .header h1 {{ font-size: 2.4rem; font-weight: 700; color: #fff; margin-bottom: 12px; }}
            .header p {{ color: var(--text-secondary); font-size: 1.1rem; }}
            .badge {{ display: inline-block; background: rgba(56, 189, 248, 0.15); color: var(--accent-blue); padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; margin-bottom: 15px; }}
            .card {{ background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
            .card h2 {{ font-size: 1.3rem; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; }}
            .btn-group {{ display: flex; gap: 12px; margin-top: 15px; }}
            .btn {{ padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 0.95rem; display: inline-flex; align-items: center; justify-content: center; }}
            .btn-primary {{ background: #2563eb; color: #fff; }}
            .btn-primary:hover {{ background: #1d4ed8; }}
            .btn-secondary {{ background: #334155; color: #fff; }}
            .btn-secondary:hover {{ background: #475569; }}
            .route-list {{ list-style: none; }}
            .route-item {{ display: flex; align-items: center; justify-content: space-between; padding: 12px; border-bottom: 1px solid var(--border-color); font-family: monospace; font-size: 0.95rem; }}
            .route-item:last-child {{ border-bottom: none; }}
            .method {{ padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.8rem; min-width: 65px; text-align: center; }}
            .method.get {{ background: rgba(74, 222, 128, 0.2); color: var(--accent-green); }}
            .method.post {{ background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); }}
            .method.put {{ background: rgba(251, 191, 36, 0.2); color: var(--accent-amber); }}
            .method.patch {{ background: rgba(192, 132, 252, 0.2); color: var(--accent-purple); }}
            .method.delete {{ background: rgba(251, 113, 133, 0.2); color: var(--accent-rose); }}
            .route-path {{ flex-grow: 1; margin: 0 15px; color: #e2e8f0; }}
            .route-desc {{ color: var(--text-secondary); font-family: sans-serif; font-size: 0.85rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <span class="badge">FASTAPI v{settings.APP_VERSION}</span>
                <h1>{settings.APP_TITLE}</h1>
                <p>{settings.APP_DESCRIPTION}</p>
                <div class="btn-group" style="justify-content: center;">
                    <a href="/docs" class="btn btn-primary">Open Interactive Swagger UI</a>
                    <a href="/redoc" class="btn btn-secondary">Open ReDoc Documentation</a>
                    <a href="/health" class="btn btn-secondary">Check /health</a>
                </div>
            </div>

            <div class="card">
                <h2><span>Supported HTTP Methods & Routes</span></h2>
                <ul class="route-list">
                    <li class="route-item">
                        <span class="method get">GET</span>
                        <span class="route-path">/health</span>
                        <span class="route-desc">Service health status</span>
                    </li>
                    <li class="route-item">
                        <span class="method get">GET</span>
                        <span class="route-path">/models</span>
                        <span class="route-desc">List all registered ML models</span>
                    </li>
                    <li class="route-item">
                        <span class="method get">GET</span>
                        <span class="route-path">/models/{{model_id}}</span>
                        <span class="route-desc">Get model details & hyperparameters</span>
                    </li>
                    <li class="route-item">
                        <span class="method post">POST</span>
                        <span class="route-path">/models</span>
                        <span class="route-desc">Register new ML model</span>
                    </li>
                    <li class="route-item">
                        <span class="method put">PUT</span>
                        <span class="route-path">/models/{{model_id}}</span>
                        <span class="route-desc">Full update / replace model config</span>
                    </li>
                    <li class="route-item">
                        <span class="method patch">PATCH</span>
                        <span class="route-path">/models/{{model_id}}</span>
                        <span class="route-desc">Partial update (active state, tags)</span>
                    </li>
                    <li class="route-item">
                        <span class="method delete">DELETE</span>
                        <span class="route-path">/models/{{model_id}}</span>
                        <span class="route-desc">Delete / unregister model</span>
                    </li>
                    <li class="route-item">
                        <span class="method post">POST</span>
                        <span class="route-path">/predict</span>
                        <span class="route-desc">Predict using default model</span>
                    </li>
                    <li class="route-item">
                        <span class="method post">POST</span>
                        <span class="route-path">/predict/{{model_id}}</span>
                        <span class="route-desc">Predict using specified model</span>
                    </li>
                    <li class="route-item">
                        <span class="method post">POST</span>
                        <span class="route-path">/predict/{{model_id}}/batch</span>
                        <span class="route-desc">Batch prediction inference</span>
                    </li>
                    <li class="route-item">
                        <span class="method get">GET</span>
                        <span class="route-path">/predict/history</span>
                        <span class="route-desc">Retrieve past prediction logs</span>
                    </li>
                    <li class="route-item">
                        <span class="method delete">DELETE</span>
                        <span class="route-path">/predict/history</span>
                        <span class="route-desc">Clear prediction history</span>
                    </li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
