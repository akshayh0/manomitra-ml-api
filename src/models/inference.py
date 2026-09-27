"""Production inference pipeline for Cognitive Distortion Classification.

Loads the champion semantic model (all-MiniLM-L6-v2 + LogisticRegression)
and provides safe, validated inference with calibrated probability rankings.

DISCLAIMER:
This module performs cognitive distortion classification for psychoeducational
and reflective assistance within Manomitra. It is NOT a diagnostic tool and does
not provide clinical or medical diagnoses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final" / "final_semantic_model.joblib"
FALLBACK_EXP_MODEL_PATH = PROJECT_ROOT / "models" / "experiments" / "exp005a_sentence_embeddings_logistic_regression.joblib"

MAX_TEXT_LENGTH = 5000  # Maximum accepted character length for input inquiry
DEFAULT_TOP_K = 3


class CognitiveDistortionClassifier:
    """Production inference wrapper for cognitive distortion classification."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the classifier and load model artifacts.

        Parameters
        ----------
        model_path : str | Path | None, optional
            Path to the saved final model joblib artifact.
        device : str, default "cpu"
            Device for sentence transformer inference ("cpu" or "cuda").
        """
        self.device = device
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not self.model_path.exists():
            if FALLBACK_EXP_MODEL_PATH.exists():
                logger.warning(
                    "Final model not found at %s. Falling back to %s",
                    self.model_path,
                    FALLBACK_EXP_MODEL_PATH,
                )
                self.model_path = FALLBACK_EXP_MODEL_PATH
            else:
                raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")

        logger.info("Loading model artifact from: %s", self.model_path)
        self.artifact = joblib.load(self.model_path)

        self.classifier = self.artifact["classifier"]
        self.classes: list[str] = list(self.classifier.classes_)
        self.encoder_name: str = self.artifact.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim: int = self.artifact.get("embedding_dim", 384)
        self.model_name: str = self.artifact.get("model_name", "all-MiniLM-L6-v2 + LogisticRegression")

        logger.info("Loading sentence encoder: %s on device: %s", self.encoder_name, self.device)
        self.encoder = SentenceTransformer(self.encoder_name, device=self.device)
        logger.info("CognitiveDistortionClassifier initialized successfully with %d classes.", len(self.classes))

    def _validate_input(self, text: Any, top_k: Any) -> tuple[str, int]:
        """Validate raw text inquiry and top_k parameter.

        Raises
        ------
        TypeError
            If text is not a string or top_k is not an integer.
        ValueError
            If text is empty, whitespace-only, or top_k is out of range.
        """
        if text is None:
            raise ValueError("Input text cannot be None. Please provide a non-empty string.")

        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}.")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if len(cleaned_text) > MAX_TEXT_LENGTH:
            logger.warning(
                "Input text length (%d chars) exceeds MAX_TEXT_LENGTH (%d chars). Truncating.",
                len(cleaned_text),
                MAX_TEXT_LENGTH,
            )
            cleaned_text = cleaned_text[:MAX_TEXT_LENGTH].strip()

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")

        num_classes = len(self.classes)
        if top_k < 1 or top_k > num_classes:
            raise ValueError(
                f"top_k must be an integer between 1 and {num_classes} (number of available classes), got {top_k}."
            )

        return cleaned_text, top_k

    def predict(self, text: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """Predict cognitive distortion categories with calibrated confidence scores.

        Parameters
        ----------
        text : str
            Raw patient question or emotional reflection text.
        top_k : int, default 3
            Number of top-ranked predictions to return (between 1 and 11).

        Returns
        -------
        dict[str, Any]
            Dictionary containing sorted prediction list with label and confidence,
            input text snippet, and medical disclaimer.
        """
        cleaned_text, valid_top_k = self._validate_input(text, top_k)

        # 1. Generate dense embedding (384-d, L2 normalized)
        embedding = self.encoder.encode(
            [cleaned_text],
            show_progress_bar=False,
            normalize_embeddings=True,
            device=self.device,
        )

        # 2. Compute probabilities across all 11 classes via Logistic Regression
        probabilities = self.classifier.predict_proba(embedding)[0]

        # 3. Sort by probability descending
        sorted_indices = np.argsort(probabilities)[::-1]

        # 4. Extract Top-K predictions
        predictions = []
        for idx in sorted_indices[:valid_top_k]:
            predictions.append(
                {
                    "label": self.classes[idx],
                    "confidence": round(float(probabilities[idx]), 4),
                }
            )

        return {
            "predictions": predictions,
            "top_k": valid_top_k,
            "input_text": cleaned_text if len(cleaned_text) <= 120 else cleaned_text[:117] + "...",
            "model": self.model_name,
            "disclaimer": (
                "These predictions reflect model confidence for psychoeducational guidance "
                "and self-reflection. They DO NOT constitute a clinical diagnosis or medical evaluation."
            ),
        }

    def predict_batch(self, texts: list[str], top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Run batch inference over multiple text inquiries."""
        if not isinstance(texts, list):
            raise TypeError(f"Expected a list of strings, got {type(texts).__name__}.")
        return [self.predict(t, top_k=top_k) for t in texts]


# Module-level singleton instance for zero-overhead repeated inference
import threading

_GLOBAL_CLASSIFIER: CognitiveDistortionClassifier | None = None
_INIT_LOCK = threading.Lock()


def get_classifier(model_path: str | Path | None = None, device: str = "cpu") -> CognitiveDistortionClassifier:
    """Get or initialize the global singleton classifier instance with thread-safe locking."""
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        with _INIT_LOCK:
            if _GLOBAL_CLASSIFIER is None:
                _GLOBAL_CLASSIFIER = CognitiveDistortionClassifier(model_path=model_path, device=device)
    return _GLOBAL_CLASSIFIER


def predict_distortion(text: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Top-level convenience function for cognitive distortion prediction.

    Parameters
    ----------
    text : str
        The input patient text inquiry.
    top_k : int, default 3
        Number of top-ranked predictions to return (1 to 11).

    Returns
    -------
    dict[str, Any]
        Structured prediction output with ranked labels and confidence values.
    """
    classifier = get_classifier()
    return classifier.predict(text=text, top_k=top_k)
