"""Pydantic schemas for the Manomitra-ML REST API.

Defines strict request/response data contracts and validation rules.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TEXT_LENGTH = 5000


class PredictRequest(BaseModel):
    """Request payload for cognitive distortion prediction."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        description="Patient inquiry, emotional reflection, or journal text to analyze.",
        examples=["I failed one exam, so I will fail at everything."],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=11,
        description="Number of top-ranked cognitive distortion categories to return (between 1 and 11).",
        examples=[3],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("Field 'text' must be a valid string.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field 'text' cannot be empty or contain only whitespace.")
        if len(cleaned) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Field 'text' length ({len(cleaned)} characters) exceeds maximum allowable length "
                f"of {MAX_TEXT_LENGTH} characters. Please provide a concise entry."
            )
        return cleaned


class PredictionItem(BaseModel):
    """Individual cognitive distortion prediction with calibrated confidence."""

    label: str = Field(
        ...,
        description="Cognitive distortion category name or 'No Distortion'.",
        examples=["Overgeneralization"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence probability for this category (sums to 1.0 across all 11 classes).",
        examples=[0.2134],
    )


class PredictResponse(BaseModel):
    """Structured response payload containing ranked predictions."""

    predictions: list[PredictionItem] = Field(
        ...,
        description="List of top-K predicted cognitive distortion categories sorted by confidence descending.",
    )
    top_k: int = Field(
        ...,
        description="The number of predictions returned.",
        examples=[3],
    )
    model: str = Field(
        ...,
        description="Champion model architecture used for inference.",
        examples=["all-MiniLM-L6-v2 + LogisticRegression"],
    )
    disclaimer: str = Field(
        ...,
        description="Medical and psychoeducational regulatory disclaimer.",
        examples=[
            "These predictions are for psychoeducational guidance and self-reflection and "
            "do not constitute a clinical diagnosis or medical evaluation."
        ],
    )


class HealthResponse(BaseModel):
    """Service health check response payload."""

    status: str = Field(
        default="healthy",
        description="Current health status of the service ('healthy').",
        examples=["healthy"],
    )
    service: str = Field(
        default="Manomitra Cognitive Distortion API",
        description="Name of the machine learning microservice.",
        examples=["Manomitra Cognitive Distortion API"],
    )
    model: str = Field(
        default="all-MiniLM-L6-v2 + LogisticRegression",
        description="Loaded cognitive distortion classification model.",
        examples=["all-MiniLM-L6-v2 + LogisticRegression"],
    )
    readiness: str = Field(
        default="ready",
        description="Model memory readiness: 'ready' if model is loaded into memory, 'not_ready' otherwise.",
        examples=["ready"],
    )
    model_loaded: bool = Field(
        default=True,
        description="Flag indicating whether the model artifact and encoder are successfully loaded in memory.",
        examples=[True],
    )


class ModelInfoResponse(BaseModel):
    """Model architecture and cognitive distortion classes metadata response."""

    model_name: str = Field(
        ...,
        description="Champion model architecture name.",
        examples=["all-MiniLM-L6-v2 + LogisticRegression"],
    )
    encoder_name: str = Field(
        ...,
        description="HuggingFace sentence transformer encoder identifier.",
        examples=["sentence-transformers/all-MiniLM-L6-v2"],
    )
    embedding_dimension: int = Field(
        ...,
        description="Dense representation embedding vector dimension.",
        examples=[384],
    )
    num_classes: int = Field(
        ...,
        description="Total number of supported cognitive distortion target classes.",
        examples=[11],
    )
    classes: list[str] = Field(
        ...,
        description="List of all 11 supported cognitive distortion class names.",
    )
    top_k_limits: dict[str, int] = Field(
        default={"min": 1, "max": 11, "default": 3},
        description="Allowable top_k parameter boundary configuration.",
    )
    disclaimer: str = Field(
        ...,
        description="Clinical psychoeducational disclaimer.",
    )
