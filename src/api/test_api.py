"""Comprehensive API test suite for the Manomitra-ML REST API.

Tests health check, valid predictions across top_k bounds, model info, docs,
and all invalid/edge-case scenarios.

CRITICAL:
Uses synthetic demonstration inputs only.
DOES NOT touch, load, or evaluate data/processed/test.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app
from src.models.inference import get_classifier


def test_health_endpoint(client: TestClient) -> None:
    """Test 1: Verify GET /health endpoint returns healthy status without requiring loaded model."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Manomitra Cognitive Distortion API"
    assert data["model"] == "all-MiniLM-L6-v2 + LogisticRegression"
    print(f"  [PASS] Test 1: GET /health returned 200 OK (service='Manomitra Cognitive Distortion API', readiness='{data.get('readiness')}').")


def test_health_after_load(client: TestClient) -> None:
    """Test 1b: Verify GET /health reports model_loaded=True after lazy initialization."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["readiness"] == "ready"
    print("  [PASS] Test 1b: GET /health confirmed model_loaded=True and readiness='ready' after prediction.")


def test_valid_prediction(client: TestClient) -> None:
    """Test 2: Verify POST /api/v1/predict with standard synthetic input."""
    payload = {
        "text": "I made one mistake, so I am a complete failure.",
        "top_k": 3,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3
    assert data["top_k"] == 3
    assert "disclaimer" in data
    assert "model" in data

    # Check predictions are sorted descending
    confs = [p["confidence"] for p in data["predictions"]]
    assert confs[0] >= confs[1] >= confs[2]
    assert all(0.0 <= c <= 1.0 for c in confs)
    print(f"  [PASS] Test 2: Valid prediction returned top-3 ({data['predictions'][0]['label']}: {confs[0]}).")


def test_top_k_bounds(client: TestClient) -> None:
    """Test 3, 4, 5: Verify boundary top_k values (1, 3, 11)."""
    text = "I feel like I failed everyone because I did not reply immediately."

    # top_k = 1
    resp_1 = client.post("/api/v1/predict", json={"text": text, "top_k": 1})
    assert resp_1.status_code == 200
    assert len(resp_1.json()["predictions"]) == 1
    print("  [PASS] Test 3: top_k = 1 returned exactly 1 prediction.")

    # top_k = 3 (default)
    resp_3 = client.post("/api/v1/predict", json={"text": text, "top_k": 3})
    assert resp_3.status_code == 200
    assert len(resp_3.json()["predictions"]) == 3
    print("  [PASS] Test 4: top_k = 3 returned exactly 3 predictions.")

    # top_k = 11 (all classes)
    resp_11 = client.post("/api/v1/predict", json={"text": text, "top_k": 11})
    assert resp_11.status_code == 200
    assert len(resp_11.json()["predictions"]) == 11
    print("  [PASS] Test 5: top_k = 11 returned all 11 classes.")


def test_empty_and_whitespace_text(client: TestClient) -> None:
    """Test 6 & 7: Verify empty and whitespace-only text inquiries are rejected."""
    # Empty text
    resp_empty = client.post("/api/v1/predict", json={"text": "", "top_k": 3})
    assert resp_empty.status_code in [400, 422], f"Expected 400/422, got {resp_empty.status_code}"
    print(f"  [PASS] Test 6: Empty text rejected with status {resp_empty.status_code}.")

    # Whitespace-only text
    resp_ws = client.post("/api/v1/predict", json={"text": "   \n\t   ", "top_k": 3})
    assert resp_ws.status_code in [400, 422], f"Expected 400/422, got {resp_ws.status_code}"
    print(f"  [PASS] Test 7: Whitespace text rejected with status {resp_ws.status_code}.")


def test_missing_text_field(client: TestClient) -> None:
    """Test 8: Verify missing text field is rejected by Pydantic."""
    resp_missing = client.post("/api/v1/predict", json={"top_k": 3})
    assert resp_missing.status_code == 422, f"Expected 422, got {resp_missing.status_code}"
    print("  [PASS] Test 8: Missing text field rejected with 422 Unprocessable Entity.")


def test_wrong_text_type(client: TestClient) -> None:
    """Test 9: Verify non-string text field is rejected with 422."""
    resp_num = client.post("/api/v1/predict", json={"text": 12345, "top_k": 3})
    assert resp_num.status_code == 422, f"Expected 422, got {resp_num.status_code}"
    print("  [PASS] Test 9: Wrong text type (int) rejected with 422 Unprocessable Entity.")


def test_invalid_top_k(client: TestClient) -> None:
    """Test 10 & 11: Verify invalid top_k values (< 1, > 11, non-int)."""
    text = "Valid demonstration inquiry text."

    # top_k = 0
    resp_0 = client.post("/api/v1/predict", json={"text": text, "top_k": 0})
    assert resp_0.status_code in [400, 422]
    print(f"  [PASS] Test 10a: top_k = 0 rejected with status {resp_0.status_code}.")

    # top_k = -5
    resp_neg = client.post("/api/v1/predict", json={"text": text, "top_k": -5})
    assert resp_neg.status_code in [400, 422]
    print(f"  [PASS] Test 10b: Negative top_k rejected with status {resp_neg.status_code}.")

    # top_k = 15 (> 11)
    resp_15 = client.post("/api/v1/predict", json={"text": text, "top_k": 15})
    assert resp_15.status_code in [400, 422]
    print(f"  [PASS] Test 11a: top_k > 11 rejected with status {resp_15.status_code}.")

    # non-integer top_k
    resp_str = client.post("/api/v1/predict", json={"text": text, "top_k": "invalid"})
    assert resp_str.status_code == 422
    print("  [PASS] Test 11b: Non-integer top_k rejected with 422.")


def test_excessive_length(client: TestClient) -> None:
    """Test 12: Verify text exceeding 5,000 characters is rejected (not silently truncated)."""
    long_text = "word " * 1200  # 6,000 chars
    resp_long = client.post("/api/v1/predict", json={"text": long_text, "top_k": 3})
    assert resp_long.status_code in [400, 422], f"Expected 400/422, got {resp_long.status_code}"
    print("  [PASS] Test 12: Text exceeding 5,000 characters rejected with status 422 (no silent truncation).")


def test_model_reuse_consecutive_requests(client: TestClient) -> None:
    """Test 13: Verify multiple consecutive requests reuse the identical cached model instance."""
    inst1 = get_classifier()
    inst2 = get_classifier()
    assert inst1 is inst2, "get_classifier() created separate instances!"

    for i in range(5):
        resp = client.post(
            "/api/v1/predict",
            json={"text": f"Synthetic test thought iteration number {i}.", "top_k": 2},
        )
        assert resp.status_code == 200

    inst3 = get_classifier()
    assert inst1 is inst3, "Classifier singleton instance mutated across requests!"
    print("  [PASS] Test 13: Multiple consecutive requests confirmed reusing identical in-memory model singleton.")


def test_model_info_and_docs(client: TestClient) -> None:
    """Test 14: Verify GET /api/v1/model-info, /docs, and /openapi.json."""
    resp_info = client.get("/api/v1/model-info")
    assert resp_info.status_code == 200
    info_data = resp_info.json()
    assert info_data["num_classes"] == 11
    assert len(info_data["classes"]) == 11
    assert info_data["embedding_dimension"] == 384
    print("  [PASS] Test 14a: GET /api/v1/model-info returned 200 with 11 classes and 384-d metadata.")

    resp_docs = client.get("/docs")
    assert resp_docs.status_code == 200
    print("  [PASS] Test 14b: GET /docs (Swagger UI) returned 200 OK.")

    resp_openapi = client.get("/openapi.json")
    assert resp_openapi.status_code == 200
    assert "paths" in resp_openapi.json()
    print("  [PASS] Test 14c: GET /openapi.json returned 200 OK.")


def main() -> None:
    print("=" * 70)
    print("EXECUTING MANOMITRA-ML API INTEGRATION & VALIDATION TEST SUITE")
    print("=" * 70)

    with TestClient(app) as test_client:
        test_health_endpoint(test_client)
        test_valid_prediction(test_client)
        test_health_after_load(test_client)
        test_top_k_bounds(test_client)
        test_empty_and_whitespace_text(test_client)
        test_missing_text_field(test_client)
        test_wrong_text_type(test_client)
        test_invalid_top_k(test_client)
        test_excessive_length(test_client)
        test_model_reuse_consecutive_requests(test_client)
        test_model_info_and_docs(test_client)

    print("=" * 70)
    print("ALL API INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
