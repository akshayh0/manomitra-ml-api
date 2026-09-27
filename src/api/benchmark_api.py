"""API Latency Benchmark for the Manomitra-ML Production REST API.

Measures and compares:
1. HTTP/API end-to-end latency (cold-start vs. warm requests via TestClient)
2. Isolated ML model inference latency (SentenceTransformer embedding + LogisticRegression)
3. HTTP middleware, validation, and JSON serialization overhead

CRITICAL:
Uses purely handcrafted synthetic text prompts.
DOES NOT touch or load data/processed/test.csv.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app
from src.models.inference import get_classifier


def run_api_benchmark(num_warm_iterations: int = 30) -> dict[str, float]:
    print("=" * 70)
    print("STARTING MANOMITRA-ML API LATENCY BENCHMARK (SYNTHETIC INPUT)")
    print("=" * 70)

    sample_text = (
        "I made one mistake on my assignment today, so I feel like I am a complete "
        "failure and nothing I ever do will be successful."
    )
    payload = {"text": sample_text, "top_k": 3}

    with TestClient(app) as client:
        # Measure cold start (first HTTP request through API pipeline)
        t_cold_start = time.perf_counter()
        resp_cold = client.post("/api/v1/predict", json=payload)
        cold_latency_ms = (time.perf_counter() - t_cold_start) * 1000.0
        assert resp_cold.status_code == 200, f"Cold request failed: {resp_cold.text}"

        print(f"\n[1] First Request / Cold Latency: {cold_latency_ms:.2f} ms")

        # Warm requests
        print(f"\n[2] Executing {num_warm_iterations} warm HTTP requests...")
        http_latencies: list[float] = []
        for _ in range(num_warm_iterations):
            t0 = time.perf_counter()
            resp = client.post("/api/v1/predict", json=payload)
            dt = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            http_latencies.append(dt)

        avg_http_ms = statistics.mean(http_latencies)
        min_http_ms = min(http_latencies)
        max_http_ms = max(http_latencies)
        med_http_ms = statistics.median(http_latencies)

        print(f"  -> Average Warm HTTP Latency: {avg_http_ms:.2f} ms")
        print(f"  -> Median Warm HTTP Latency:  {med_http_ms:.2f} ms")
        print(f"  -> Minimum Warm HTTP Latency: {min_http_ms:.2f} ms")
        print(f"  -> Maximum Warm HTTP Latency: {max_http_ms:.2f} ms")

        # Isolated ML inference latency (pure Python/PyTorch/scikit-learn without HTTP stack)
        print(f"\n[3] Measuring isolated ML model inference latency ({num_warm_iterations} iterations)...")
        classifier = get_classifier()
        model_latencies: list[float] = []
        for _ in range(num_warm_iterations):
            t0 = time.perf_counter()
            _ = classifier.predict(text=sample_text, top_k=3)
            dt = (time.perf_counter() - t0) * 1000.0
            model_latencies.append(dt)

        avg_model_ms = statistics.mean(model_latencies)
        min_model_ms = min(model_latencies)
        max_model_ms = max(model_latencies)

        print(f"  -> Average Isolated Model Latency: {avg_model_ms:.2f} ms")
        print(f"  -> Minimum Isolated Model Latency: {min_model_ms:.2f} ms")
        print(f"  -> Maximum Isolated Model Latency: {max_model_ms:.2f} ms")

        # Overhead breakdown
        http_overhead_ms = max(0.0, avg_http_ms - avg_model_ms)
        print(f"\n[4] Architectural Latency Breakdown:")
        print(f"  - Isolated ML Inference:   {avg_model_ms:6.2f} ms ({(avg_model_ms / avg_http_ms) * 100:5.1f}%)")
        print(f"  - HTTP/FastAPI Overhead:   {http_overhead_ms:6.2f} ms ({(http_overhead_ms / avg_http_ms) * 100:5.1f}%)")
        print(f"  - Total End-to-End API:    {avg_http_ms:6.2f} ms (100.0%)")
        print(f"  - Est. Max Single-Core QPS:{1000.0 / avg_http_ms:6.1f} requests/sec")

        print("=" * 70)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 70)

        return {
            "cold_latency_ms": cold_latency_ms,
            "avg_http_ms": avg_http_ms,
            "median_http_ms": med_http_ms,
            "min_http_ms": min_http_ms,
            "max_http_ms": max_http_ms,
            "avg_model_ms": avg_model_ms,
            "min_model_ms": min_model_ms,
            "max_model_ms": max_model_ms,
            "http_overhead_ms": http_overhead_ms,
            "qps": 1000.0 / avg_http_ms,
        }


if __name__ == "__main__":
    run_api_benchmark()
