"""Production inference pipeline for Cognitive Distortion Classification.

Uses a local INT8 ONNX all-MiniLM-L6-v2 encoder plus the existing
LogisticRegression classifier artifact.

DISCLAIMER:
This module performs cognitive distortion classification for psychoeducational
and reflective assistance within Manomitra. It is NOT a diagnostic tool and does
not provide clinical or medical diagnoses.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

# Prevent transformers/torch from eagerly allocating PyTorch memory if imported
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"

import joblib
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "final" / "final_semantic_model.joblib"
)

FALLBACK_EXP_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "experiments"
    / "exp005a_sentence_embeddings_logistic_regression.joblib"
)

ONNX_ENCODER_DIR = PROJECT_ROOT / "models" / "onnx_encoder"
ONNX_MODEL_FILE = "model_int8.onnx"

MAX_TEXT_LENGTH = 5000
DEFAULT_TOP_K = 3


class CognitiveDistortionClassifier:
    """Production inference wrapper using direct ONNX Runtime with INT8 MiniLM."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
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
                raise FileNotFoundError(
                    f"Model artifact not found at: {self.model_path}"
                )

        if not ONNX_ENCODER_DIR.exists():
            raise FileNotFoundError(
                f"ONNX encoder directory not found at: {ONNX_ENCODER_DIR}"
            )

        onnx_model_path = ONNX_ENCODER_DIR / ONNX_MODEL_FILE

        if not onnx_model_path.exists():
            raise FileNotFoundError(
                f"INT8 ONNX model not found at: {onnx_model_path}"
            )

        logger.info("Loading classifier artifact from: %s", self.model_path)
        load_start = time.perf_counter()

        artifact = joblib.load(self.model_path)
        self.classifier = artifact["classifier"]
        self.classes: list[str] = list(self.classifier.classes_)

        self.encoder_name: str = artifact.get(
            "encoder_name",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.embedding_dim: int = artifact.get(
            "embedding_dim",
            384,
        )

        self.model_name: str = artifact.get(
            "model_name",
            "all-MiniLM-L6-v2 + LogisticRegression",
        )

        # Free temporary artifact dictionary
        del artifact
        gc.collect()

        logger.info(
            "Loading local INT8 ONNX encoder from: %s",
            onnx_model_path,
        )

        # Configure low-memory ONNX Runtime SessionOptions for Render 512MB RAM
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ONNX_ENCODER_DIR),
            local_files_only=True,
        )

        load_elapsed = time.perf_counter() - load_start
        logger.info(
            "CognitiveDistortionClassifier initialized successfully "
            "in %.2f seconds with %d classes and %d-d embeddings.",
            load_elapsed,
            len(self.classes),
            self.embedding_dim,
        )

    def _validate_input(
        self,
        text: Any,
        top_k: Any,
    ) -> tuple[str, int]:

        if text is None:
            raise ValueError(
                "Input text cannot be None. Please provide a non-empty string."
            )

        if not isinstance(text, str):
            raise TypeError(
                f"Input text must be a string, got {type(text).__name__}."
            )

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError(
                "Input text cannot be empty or whitespace-only."
            )

        if len(cleaned_text) > MAX_TEXT_LENGTH:
            logger.warning(
                "Input text length (%d chars) exceeds MAX_TEXT_LENGTH (%d chars). Truncating.",
                len(cleaned_text),
                MAX_TEXT_LENGTH,
            )
            cleaned_text = cleaned_text[:MAX_TEXT_LENGTH].strip()

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(
                f"top_k must be an integer, got {type(top_k).__name__}."
            )

        num_classes = len(self.classes)

        if top_k < 1 or top_k > num_classes:
            raise ValueError(
                f"top_k must be an integer between 1 and {num_classes} "
                f"(number of available classes), got {top_k}."
            )

        return cleaned_text, top_k

    def _encode(self, text: str) -> np.ndarray:
        """Generate a normalized 384-dimensional sentence embedding using ONNX Runtime."""

        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in inputs:
            onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
        else:
            onnx_inputs["token_type_ids"] = np.zeros_like(
                inputs["input_ids"],
                dtype=np.int64,
            )

        outputs = self.session.run(None, onnx_inputs)[0]

        token_embeddings = np.asarray(
            outputs,
            dtype=np.float32,
        )

        attention_mask = np.asarray(
            inputs["attention_mask"],
            dtype=np.float32,
        )

        mask = attention_mask[..., None]

        # Mean pooling over non-padding tokens.
        embedding = (
            (token_embeddings * mask).sum(axis=1)
            / np.clip(mask.sum(axis=1), 1e-9, None)
        )

        # L2 normalization, matching the original
        # SentenceTransformer normalize_embeddings=True behavior.
        norm = np.linalg.norm(
            embedding,
            axis=1,
            keepdims=True,
        )

        embedding = embedding / np.clip(
            norm,
            1e-12,
            None,
        )

        if embedding.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"Unexpected embedding dimension: {embedding.shape[1]}. "
                f"Expected {self.embedding_dim}."
            )

        return embedding.astype(np.float32)

    def predict(
        self,
        text: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:

        cleaned_text, valid_top_k = self._validate_input(
            text,
            top_k,
        )

        # 1. Generate dense 384-dimensional normalized embedding.
        embedding = self._encode(cleaned_text)

        # 2. Logistic Regression probabilities.
        probabilities = self.classifier.predict_proba(embedding)[0]

        # 3. Sort probabilities descending.
        sorted_indices = np.argsort(probabilities)[::-1]

        # 4. Extract Top-K predictions.
        predictions = []

        for idx in sorted_indices[:valid_top_k]:
            predictions.append(
                {
                    "label": self.classes[idx],
                    "confidence": round(
                        float(probabilities[idx]),
                        4,
                    ),
                }
            )

        return {
            "predictions": predictions,
            "top_k": valid_top_k,
            "input_text": (
                cleaned_text
                if len(cleaned_text) <= 120
                else cleaned_text[:117] + "..."
            ),
            "model": self.model_name,
            "disclaimer": (
                "These predictions reflect model confidence for "
                "psychoeducational guidance and self-reflection. "
                "They DO NOT constitute a clinical diagnosis or medical evaluation."
            ),
        }

    def predict_batch(
        self,
        texts: list[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:

        if not isinstance(texts, list):
            raise TypeError(
                f"Expected a list of strings, got {type(texts).__name__}."
            )

        return [
            self.predict(text, top_k=top_k)
            for text in texts
        ]


_GLOBAL_CLASSIFIER: CognitiveDistortionClassifier | None = None
_INIT_LOCK = threading.Lock()


def get_classifier(
    model_path: str | Path | None = None,
    device: str = "cpu",
) -> CognitiveDistortionClassifier:

    global _GLOBAL_CLASSIFIER

    if _GLOBAL_CLASSIFIER is None:
        with _INIT_LOCK:
            if _GLOBAL_CLASSIFIER is None:
                logger.info(
                    "Initializing CognitiveDistortionClassifier on first request (lazy load)..."
                )
                t0 = time.perf_counter()
                _GLOBAL_CLASSIFIER = CognitiveDistortionClassifier(
                    model_path=model_path,
                    device=device,
                )
                elapsed = time.perf_counter() - t0
                logger.info(
                    "CognitiveDistortionClassifier loaded successfully in %.2f seconds.",
                    elapsed,
                )

    return _GLOBAL_CLASSIFIER


def predict_distortion(
    text: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:

    classifier = get_classifier()

    return classifier.predict(
        text=text,
        top_k=top_k,
    )
