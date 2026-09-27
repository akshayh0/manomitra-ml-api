"""Verification test ensuring the final model artifact is 100% PyTorch-independent.

Verifies that in a Python environment without PyTorch (torch) installed,
`joblib.load("models/final/final_semantic_model.joblib")` succeeds and can
accurately produce predictions using only numpy and scikit-learn.
"""

from __future__ import annotations

import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Enforce strict PyTorch block in sys.meta_path to simulate environment
#    where torch is completely uninstalled (e.g. Render 512MB RAM instance).
# -----------------------------------------------------------------------------
class TorchBlocker:
    """Import hook that strictly raises ModuleNotFoundError for any torch imports."""
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError(f"No module named '{fullname}' (simulated uninstalled PyTorch)")
        return None

sys.meta_path.insert(0, TorchBlocker())

# Purge any existing torch modules if loaded earlier in process
for mod_name in list(sys.modules.keys()):
    if mod_name == "torch" or mod_name.startswith("torch."):
        del sys.modules[mod_name]

# Verify torch cannot be imported
try:
    import torch
    raise RuntimeError("PyTorch blocking failed: 'import torch' succeeded!")
except ModuleNotFoundError as exc:
    print(f"  [PASS] Confirmed: PyTorch is strictly blocked ({exc}).")

# -----------------------------------------------------------------------------
# 2. Test joblib.load on models/final/final_semantic_model.joblib without torch
# -----------------------------------------------------------------------------
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "final_semantic_model.joblib"


def test_joblib_load_without_torch() -> None:
    """Verify loading the final joblib artifact without torch installed."""
    assert MODEL_PATH.exists(), f"Model artifact not found at: {MODEL_PATH}"
    print(f"Loading artifact from: {MODEL_PATH} ...")

    artifact = joblib.load(MODEL_PATH)
    assert isinstance(artifact, dict), f"Expected dict, got {type(artifact)}"
    print("  [PASS] joblib.load succeeded without importing torch.")

    # Validate essential keys
    expected_keys = [
        "model_name",
        "encoder_name",
        "embedding_dim",
        "classifier",
        "labels",
        "hyperparameters",
        "validation_metrics",
        "test_metrics",
        "random_seed",
        "environment",
    ]
    for key in expected_keys:
        assert key in artifact, f"Missing expected key: {key}"
    print(f"  [PASS] All expected keys present: {list(artifact.keys())}")

    # Validate classifier type and attributes
    clf = artifact["classifier"]
    assert isinstance(clf, LogisticRegression), f"Expected LogisticRegression, got {type(clf)}"
    assert hasattr(clf, "classes_"), "Classifier missing classes_ attribute"
    assert hasattr(clf, "coef_"), "Classifier missing coef_ attribute"
    assert hasattr(clf, "intercept_"), "Classifier missing intercept_ attribute"

    classes = list(clf.classes_)
    assert len(classes) == 11, f"Expected 11 classes, got {len(classes)}"
    assert clf.coef_.shape == (11, 384), f"Expected coef_ shape (11, 384), got {clf.coef_.shape}"
    assert clf.intercept_.shape == (11,), f"Expected intercept_ shape (11,), got {clf.intercept_.shape}"
    print(f"  [PASS] Classifier verified: 11 classes, coef shape {clf.coef_.shape}.")

    # Validate prediction execution
    dummy_embedding = np.random.randn(1, 384).astype(np.float32)
    norm = np.linalg.norm(dummy_embedding, axis=1, keepdims=True)
    dummy_embedding = dummy_embedding / np.maximum(norm, 1e-12)

    probs = clf.predict_proba(dummy_embedding)
    assert probs.shape == (1, 11), f"Expected probs shape (1, 11), got {probs.shape}"
    prob_sum = float(np.sum(probs))
    assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities do not sum to 1.0: {prob_sum}"
    print(f"  [PASS] Probability prediction succeeded (sum={prob_sum:.4f}, classes={len(probs[0])}).")

    # Validate environment metadata contains only clean types (no custom torch classes)
    env = artifact.get("environment", {})
    for k, v in env.items():
        assert type(v) is str, f"Environment value for '{k}' must be str, got {type(v)}"
        assert "TorchVersion" not in str(type(v)), f"TorchVersion leaked into '{k}'"
    print(f"  [PASS] Environment metadata contains pure strings: {env}")

    # Final assertion: ensure torch was NEVER loaded into sys.modules
    assert "torch" not in sys.modules, "torch was unexpectedly imported into sys.modules!"
    for m in sys.modules:
        assert not m.startswith("torch."), f"torch submodule '{m}' was unexpectedly imported!"
    print("  [PASS] sys.modules verified: zero torch modules loaded.")


if __name__ == "__main__":
    print("=" * 70)
    print("VERIFYING PYTORCH-INDEPENDENT JOBLIB DESERIALIZATION & PREDICTION")
    print("=" * 70)
    test_joblib_load_without_torch()
    print("=" * 70)
    print("ALL PYTORCH-INDEPENDENCE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
