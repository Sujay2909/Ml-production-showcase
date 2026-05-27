"""
FastAPI application for real-time ML model scoring.

Endpoints
---------
POST /predict/churn       – customer churn probability
POST /predict/demand      – demand forecast
POST /classify/document   – NLP document classification
POST /extract/entities    – NLP entity extraction
GET  /health              – liveness probe
GET  /metrics             – cache + model stats
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.logger import get_logger
from src.settings import get_settings

from .cache import RedisCache

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class CustomerFeaturePayload(BaseModel):
    customer_id: str
    total_events: int = Field(..., ge=0)
    unique_products: int = Field(..., ge=0)
    total_revenue: float = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)
    avg_session_sec: float = Field(..., ge=0)
    days_since_last_event: int = Field(..., ge=0)
    customer_tenure_days: int = Field(..., ge=0)
    active_days: int = Field(..., ge=0)
    engagement_score: float = Field(..., ge=0, le=1)


class DemandPayload(BaseModel):
    product_id: str
    region: str
    daily_units: float
    daily_revenue: float
    daily_unique_buyers: int
    revenue_rolling_7d: Optional[float] = None
    units_rolling_7d: Optional[float] = None
    revenue_rolling_14d: Optional[float] = None
    units_rolling_14d: Optional[float] = None
    revenue_rolling_30d: Optional[float] = None
    units_rolling_30d: Optional[float] = None


class DocumentPayload(BaseModel):
    doc_id: str = "doc"
    text: str = Field(..., min_length=10)


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    prediction: Any
    churn_probability: Optional[float] = None
    cached: bool = False
    latency_ms: float


class DocumentClassificationResponse(BaseModel):
    doc_id: str
    label: str
    confidence: float
    requires_review: bool
    all_scores: Dict[str, float]
    cached: bool = False
    latency_ms: float


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    cache: Optional[RedisCache] = None,
    model_churn=None,
    model_demand=None,
    nlp_classifier=None,
    nlp_extractor=None,
) -> FastAPI:
    """
    Factory function — accepts injected dependencies for testability.
    In production, models are loaded from MODEL_REGISTRY_PATH.
    """
    settings = get_settings()
    _cache = cache or RedisCache()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("api_startup", env=settings.app_env)
        yield
        logger.info("api_shutdown")

    app = FastAPI(
        title="ML Platform API",
        description="Real-time ML scoring: churn prediction, demand forecasting, NLP classification",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------
    # Request timing middleware
    # ----------------------------------------------------------------
    @app.middleware("http")
    async def add_latency_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
        return response

    # ----------------------------------------------------------------
    # Health
    # ----------------------------------------------------------------
    @app.get("/health", tags=["ops"])
    async def health():
        return {"status": "ok", "env": settings.app_env}

    @app.get("/metrics", tags=["ops"])
    async def metrics():
        return {"cache": _cache.stats()}

    # ----------------------------------------------------------------
    # Churn prediction
    # ----------------------------------------------------------------
    @app.post("/predict/churn", response_model=PredictionResponse, tags=["ml"])
    async def predict_churn(payload: CustomerFeaturePayload):
        t0 = time.perf_counter()
        cache_key = RedisCache.make_key("churn", payload.model_dump())

        cached_result = _cache.get(cache_key)
        if cached_result:
            return PredictionResponse(
                customer_id=payload.customer_id,
                prediction=cached_result["prediction"],
                churn_probability=cached_result.get("churn_probability"),
                cached=True,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        if model_churn is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Churn model not loaded",
            )

        import pandas as pd

        df = pd.DataFrame([payload.model_dump(exclude={"customer_id"})])
        result = model_churn.predict(df)
        churn_prob = (
            result.get("churn_probability", [None])[0] if "churn_probability" in result else None
        )
        prediction = result["predictions"][0]

        _cache.set(cache_key, {"prediction": prediction, "churn_probability": churn_prob})
        logger.info(
            "churn_prediction",
            customer_id=payload.customer_id,
            prediction=prediction,
            churn_probability=churn_prob,
        )
        return PredictionResponse(
            customer_id=payload.customer_id,
            prediction=prediction,
            churn_probability=churn_prob,
            cached=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ----------------------------------------------------------------
    # Document classification
    # ----------------------------------------------------------------
    @app.post(
        "/classify/document",
        response_model=DocumentClassificationResponse,
        tags=["nlp"],
    )
    async def classify_document(payload: DocumentPayload):
        t0 = time.perf_counter()
        cache_key = RedisCache.make_key("nlp_clf", payload.model_dump())

        cached = _cache.get(cache_key)
        if cached:
            return DocumentClassificationResponse(
                **cached,
                cached=True,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        if nlp_classifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NLP classifier not loaded",
            )

        clf_result = nlp_classifier.classify(payload.text)
        response_data = {
            "doc_id": payload.doc_id,
            "label": clf_result.label,
            "confidence": clf_result.confidence,
            "requires_review": clf_result.requires_review,
            "all_scores": clf_result.all_scores,
        }
        _cache.set(cache_key, response_data)
        return DocumentClassificationResponse(
            **response_data,
            cached=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ----------------------------------------------------------------
    # Entity extraction
    # ----------------------------------------------------------------
    @app.post("/extract/entities", tags=["nlp"])
    async def extract_entities(payload: DocumentPayload):
        if nlp_extractor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Entity extractor not loaded",
            )
        result = nlp_extractor.extract(payload.text, doc_id=payload.doc_id)
        return result.to_dict()

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

# Module-level app instance used by uvicorn / gunicorn.
# RedisCache now connects lazily so this is safe to import without Redis running.
app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.serving.api:app",
        host=settings.serving_host,
        port=settings.serving_port,
        workers=4,
        reload=settings.app_env == "development",
    )
