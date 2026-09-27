"""Standalone script to verify running live uvicorn server locally."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def verify_live_server() -> None:
    print(f"Launching live uvicorn process via {VENV_PYTHON} on port 8088...")
    proc = subprocess.Popen(
        [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8088",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    try:
        started = False
        print("Waiting for server startup on http://127.0.0.1:8088 ...")
        for _ in range(40):
            time.sleep(0.5)
            try:
                resp = httpx.get("http://127.0.0.1:8088/health", timeout=1.0)
                if resp.status_code == 200:
                    started = True
                    print(f"Server is LIVE! Status code: {resp.status_code}")
                    print("GET /health ->", resp.json())
                    break
            except Exception:
                pass

        if not started:
            raise RuntimeError("Live server failed to bind and respond within 20s.")

        # Test GET /api/v1/model-info
        info_resp = httpx.get("http://127.0.0.1:8088/api/v1/model-info", timeout=5.0)
        assert info_resp.status_code == 200
        print("GET /api/v1/model-info -> Classes:", len(info_resp.json()["classes"]))

        # Test GET /docs
        docs_resp = httpx.get("http://127.0.0.1:8088/docs", timeout=5.0)
        assert docs_resp.status_code == 200
        print("GET /docs -> Status 200 OK")

        # Test GET /openapi.json
        openapi_resp = httpx.get("http://127.0.0.1:8088/openapi.json", timeout=5.0)
        assert openapi_resp.status_code == 200
        print("GET /openapi.json -> Status 200 OK")

        # Test POST /api/v1/predict
        payload = {
            "text": "I made one mistake, so I am a complete failure.",
            "top_k": 3,
        }
        t0 = time.time()
        post_resp = httpx.post("http://127.0.0.1:8088/api/v1/predict", json=payload, timeout=20.0)
        latency_ms = (time.time() - t0) * 1000.0

        print(f"\nPOST /api/v1/predict responded in {latency_ms:.2f} ms (Status: {post_resp.status_code}):")
        data = post_resp.json()
        print("Predictions:")
        for rank, p in enumerate(data["predictions"], start=1):
            print(f"  {rank}. {p['label']:25s} (Confidence: {p['confidence']:.4f})")
        print("Model:", data["model"])
        print("Disclaimer:", data["disclaimer"][:60] + "...")

    finally:
        print("\nShutting down live uvicorn server...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server shutdown complete.")


if __name__ == "__main__":
    verify_live_server()
