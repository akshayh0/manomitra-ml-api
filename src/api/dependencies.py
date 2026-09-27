"""Dependency injection providers for the Manomitra-ML API.

Manages cached access to the CognitiveDistortionClassifier singleton instance.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, status

from src.models.inference import (
    DEFAULT_MODEL_PATH,
    FALLBACK_EXP_MODEL_PATH,
    CognitiveDistortionClassifier,
    get_classifier,
)

logger = logging.getLogger(__name__)


def get_classifier_dep() -> CognitiveDistortionClassifier:
    """FastAPI dependency to retrieve the cached classifier instance.

    Raises HTTP 503 if the model is unavailable or failed to load.
    """
    try:
        return get_classifier()
    except (FileNotFoundError, RuntimeError) as err:
        logger.critical("Model initialization failed: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cognitive distortion model is currently unavailable or initializing.",
        )
    except Exception as exc:
        logger.critical("Unexpected error retrieving model: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error accessing model service.",
        )


def check_model_loaded() -> bool:
    """Check whether the model is already loaded in memory.

    Does NOT run inference or trigger expensive model loading.
    """
    import src.models.inference as inf_mod

    return inf_mod._GLOBAL_CLASSIFIER is not None
