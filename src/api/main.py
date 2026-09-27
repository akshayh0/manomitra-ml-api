"""FastAPI application entrypoint for the Manomitra-ML inference service.

Provides RESTful endpoints for health monitoring, model metadata,
and cognitive distortion prediction.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import check_model_loaded, get_classifier_dep
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.models.inference import CognitiveDistortionClassifier

# Configure logging: strictly avoid logging raw patient/user text inquiries
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("manomitra.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup initialization and graceful shutdown for Manomitra-ML API service.

    Warms up the champion model singleton during boot to prevent latency spikes
    on the first client request.
    """
    logger.info("Initializing Manomitra-ML API service and pre-warming champion model...")
    try:
        classifier = get_classifier_dep()
        logger.info(
            "Champion model '%s' successfully loaded into memory (%d classes).",
            classifier.model_name,
            len(classifier.classes),
        )
    except Exception as exc:
        logger.critical("Failed to pre-warm classifier during application startup: %s", exc)
    yield
    logger.info("Shutting down Manomitra-ML API service.")


app = FastAPI(
    title="Manomitra Cognitive Distortion Classification API",
    description=(
        "Production REST API for classifying cognitive distortions in patient inquiries "
        "and self-reflection text within Manomitra. Powered by dense sentence transformer "
        "embeddings (all-MiniLM-L6-v2) and balanced multinomial logistic regression.\n\n"
        "**CLINICAL NOTICE**: Predictions are for psychoeducational guidance and self-reflection "
        "only and do NOT constitute a medical evaluation or clinical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# -----------------------------------------------------------------------------
# CORS Configuration
# Configurable via the MANOMITRA_CORS_ORIGINS environment variable.
# For local development, safe explicit loopback origins are configured.
# Unrestricted wildcard origins (allow_origins=["*"]) are deliberately avoided.
# -----------------------------------------------------------------------------
cors_origins_env = os.getenv("MANOMITRA_CORS_ORIGINS", "")
if cors_origins_env.strip():
    ALLOWED_ORIGINS = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
else:
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


# -----------------------------------------------------------------------------
# Exception Handlers
# Clean, privacy-conscious error responses without leaking filesystem paths,
# environment variables, internal model parameters, or raw user text.
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic schema validation failures into clean API error responses."""
    errors = []
    for err in exc.errors():
        loc_str = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append(f"{loc_str}: {err.get('msg', 'Invalid input')}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unprocessable Entity",
            "message": "Input validation failed. Please check the request fields.",
            "details": errors,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle domain validation errors from the inference layer."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "message": str(exc),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected server exceptions without leaking internal stack traces."""
    logger.error("Unhandled server exception during request processing: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing the cognitive distortion inquiry.",
        },
    )


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health & Liveness Check",
    tags=["Monitoring"],
)
def health_check() -> HealthResponse:
    """Verify service liveness and model readiness without triggering inference or unnecessary model loading."""
    is_loaded = check_model_loaded()
    return HealthResponse(
        status="healthy",
        service="Manomitra Cognitive Distortion API",
        model="all-MiniLM-L6-v2 + LogisticRegression",
        readiness="ready" if is_loaded else "not_ready",
        model_loaded=is_loaded,
    )


@app.get(
    "/api/v1/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata & Class Information",
    tags=["Inference"],
)
def model_info(
    classifier: CognitiveDistortionClassifier = Depends(get_classifier_dep),
) -> ModelInfoResponse:
    """Return public model architecture metadata and supported distortion classes."""
    return ModelInfoResponse(
        model_name=classifier.model_name,
        encoder_name=classifier.encoder_name,
        embedding_dimension=classifier.embedding_dim,
        num_classes=len(classifier.classes),
        classes=classifier.classes,
        top_k_limits={"min": 1, "max": len(classifier.classes), "default": 3},
        disclaimer=(
            "These predictions are for psychoeducational guidance and self-reflection and "
            "do not constitute a clinical diagnosis or medical evaluation."
        ),
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    summary="Predict Cognitive Distortions",
    tags=["Inference"],
)
def predict(
    payload: PredictRequest,
    classifier: CognitiveDistortionClassifier = Depends(get_classifier_dep),
) -> PredictResponse:
    """Analyze a patient text inquiry and return top-K cognitive distortion predictions.

    - **text**: Raw text reflection or question to analyze (max 5,000 chars).
    - **top_k**: Number of candidate predictions to return (between 1 and 11, default 3).
    """
    try:
        raw_result = classifier.predict(text=payload.text, top_k=payload.top_k)
        return PredictResponse(
            predictions=raw_result["predictions"],
            top_k=raw_result["top_k"],
            model=raw_result["model"],
            disclaimer=(
                "These predictions are for psychoeducational guidance and self-reflection and "
                "do not constitute a clinical diagnosis or medical evaluation."
            ),
        )
    except (ValueError, TypeError) as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.error("Inference processing failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected server error occurred while processing the cognitive distortion inquiry.",
        )
