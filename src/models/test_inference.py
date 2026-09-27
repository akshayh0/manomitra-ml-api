"""Safe smoke tests and latency benchmarks for the production inference pipeline.

CRITICAL:
This script uses purely synthetic, handcrafted demonstration prompts.
It DOES NOT touch, load, or evaluate on data/processed/test.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path for direct invocation
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import (
    CognitiveDistortionClassifier,
    get_classifier,
    predict_distortion,
)


def run_smoke_tests() -> dict[str, Any]:
    print("=" * 70)
    print("STARTING INFERENCE SMOKE TESTS & VALIDATION CHECKS")
    print("=" * 70)

    # 1. Model Loading Check
    print("\n[Test 1] Initializing CognitiveDistortionClassifier...")
    t0 = time.time()
    classifier = get_classifier()
    load_time = time.time() - t0
    print(f"  -> Model and encoder loaded in {load_time:.3f} seconds.")
    print(f"  -> Model name: {classifier.model_name}")
    print(f"  -> Encoder: {classifier.encoder_name} ({classifier.embedding_dim}-d)")
    print(f"  -> Number of classes: {len(classifier.classes)}")
    assert len(classifier.classes) == 11, f"Expected 11 classes, got {len(classifier.classes)}"

    # 2. Synthetic Test Prompts (Purely synthetic, NOT from test.csv)
    synthetic_cases = [
        {
            "name": "Labeling",
            "text": "I made a simple typo in my presentation today and now I know that I am a completely incompetent fool.",
        },
        {
            "name": "All-or-nothing thinking",
            "text": "If I cannot do this job with 100% perfection, then the entire effort is a total waste and I am a failure.",
        },
        {
            "name": "Overgeneralization",
            "text": "Nobody in my life ever listens to me, and absolutely nothing good ever happens to me.",
        },
        {
            "name": "Mind reading",
            "text": "I noticed the way my coworker glanced at me during lunch, and I am certain she despises me.",
        },
        {
            "name": "Emotional reasoning",
            "text": "I feel so terribly overwhelmed and guilty today, which proves that I must be an awful person.",
        },
        {
            "name": "No Distortion (Neutral Inquiry)",
            "text": "I have been experiencing mild insomnia for three nights and would like recommendations on healthy sleep hygiene.",
        },
    ]

    print("\n[Test 2] Testing synthetic inference cases across distortion types...")
    predictions_log = []
    for case in synthetic_cases:
        result = predict_distortion(case["text"], top_k=3)
        predictions_log.append({"case": case["name"], "result": result})
        print(f"\n  Case: {case['name']}")
        print(f"  Input: \"{result['input_text']}\"")
        print("  Top-3 Predictions:")
        for rank, p in enumerate(result["predictions"], start=1):
            print(f"    {rank}. {p['label']:25s} (Confidence: {p['confidence']:.4f})")

        # Invariant checks
        assert len(result["predictions"]) == 3, "top_k=3 did not return 3 predictions"
        confidences = [p["confidence"] for p in result["predictions"]]
        assert confidences[0] >= confidences[1] >= confidences[2], "Predictions not sorted descending"
        assert all(0.0 <= c <= 1.0 for c in confidences), "Confidence out of [0, 1] range"
        assert "disclaimer" in result, "Medical disclaimer missing"

    # 3. Input Validation & Edge Case Handling
    print("\n[Test 3] Testing Input Validation & Error Handling...")

    # Case A: Empty string
    try:
        predict_distortion("")
        assert False, "Failed to reject empty string"
    except ValueError as e:
        print(f"  [PASS] Empty string correctly rejected: '{e}'")

    # Case B: Whitespace-only string
    try:
        predict_distortion("   \n\t   ")
        assert False, "Failed to reject whitespace-only string"
    except ValueError as e:
        print(f"  [PASS] Whitespace-only correctly rejected: '{e}'")

    # Case C: None input
    try:
        predict_distortion(None)  # type: ignore
        assert False, "Failed to reject None input"
    except ValueError as e:
        print(f"  [PASS] None input correctly rejected: '{e}'")

    # Case D: Non-string input
    try:
        predict_distortion(12345)  # type: ignore
        assert False, "Failed to reject numeric input"
    except TypeError as e:
        print(f"  [PASS] Non-string correctly rejected: '{e}'")

    # Case E: Invalid top_k = 0
    try:
        predict_distortion("Hello world", top_k=0)
        assert False, "Failed to reject top_k=0"
    except ValueError as e:
        print(f"  [PASS] top_k=0 correctly rejected: '{e}'")

    # Case F: Invalid top_k = 15 (> 11 classes)
    try:
        predict_distortion("Hello world", top_k=15)
        assert False, "Failed to reject top_k=15"
    except ValueError as e:
        print(f"  [PASS] top_k > 11 correctly rejected: '{e}'")

    # Case G: Non-integer top_k
    try:
        predict_distortion("Hello world", top_k="3")  # type: ignore
        assert False, "Failed to reject string top_k"
    except TypeError as e:
        print(f"  [PASS] String top_k correctly rejected: '{e}'")

    # Case H: Valid boundary top_k = 1 and top_k = 11
    res_1 = predict_distortion("I feel bad today.", top_k=1)
    assert len(res_1["predictions"]) == 1, "top_k=1 failed"
    print(f"  [PASS] Boundary top_k=1 returned exactly 1 prediction: {res_1['predictions'][0]['label']}")

    res_11 = predict_distortion("I feel bad today.", top_k=11)
    assert len(res_11["predictions"]) == 11, "top_k=11 failed"
    print(f"  [PASS] Boundary top_k=11 returned all 11 classes.")

    # 4. Latency & Performance Breakdown
    print("\n[Test 4] Benchmarking Latency on Synthetic Inputs (Warmup + 10 iterations)...")
    sample_text = "I feel like I failed everyone because I did not reply to their messages immediately."

    # Warm-up run
    classifier.predict(sample_text)

    num_iterations = 20
    embed_times = []
    clf_times = []
    total_times = []

    for _ in range(num_iterations):
        t_start = time.perf_counter()
        
        # Isolated embedding latency
        t_e0 = time.perf_counter()
        emb = classifier.encoder.encode([sample_text], show_progress_bar=False, normalize_embeddings=True, device="cpu")
        t_embed = time.perf_counter() - t_e0
        embed_times.append(t_embed)

        # Isolated classifier latency
        t_c0 = time.perf_counter()
        _ = classifier.classifier.predict_proba(emb)
        t_clf = time.perf_counter() - t_c0
        clf_times.append(t_clf)

        t_total = time.perf_counter() - t_start
        total_times.append(t_total)

    avg_embed_ms = (sum(embed_times) / num_iterations) * 1000.0
    avg_clf_ms = (sum(clf_times) / num_iterations) * 1000.0
    avg_total_ms = (sum(total_times) / num_iterations) * 1000.0

    print(f"  -> Model Loading Latency:       {load_time * 1000.0:.2f} ms")
    print(f"  -> Average Embedding Latency:   {avg_embed_ms:.2f} ms")
    print(f"  -> Average Classifier Latency:  {avg_clf_ms:.2f} ms")
    print(f"  -> Average Total End-to-End:    {avg_total_ms:.2f} ms")
    print(f"  -> Estimated Single-Core QPS:   {1000.0 / avg_total_ms:.1f} requests/sec")

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS AND VALIDATIONS PASSED SUCCESSFULLY")
    print("=" * 70)

    return {
        "load_time_ms": load_time * 1000.0,
        "avg_embed_ms": avg_embed_ms,
        "avg_clf_ms": avg_clf_ms,
        "avg_total_ms": avg_total_ms,
        "qps": 1000.0 / avg_total_ms,
        "predictions_log": predictions_log,
    }


if __name__ == "__main__":
    run_smoke_tests()
