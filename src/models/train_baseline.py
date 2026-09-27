"""Train and evaluate a reproducible TF-IDF + LinearSVC baseline.

The existing train/validation/test CSVs are treated as fixed inputs. The test
split is evaluated once after selecting ``C`` on validation macro-F1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
TEXT_COLUMN = "Patient Question"
TARGET_COLUMN = "Dominant Distortion"
RANDOM_SEED = 42
C_VALUES = (0.5, 1.0, 2.0)


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {TEXT_COLUMN, TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame[[TEXT_COLUMN, TARGET_COLUMN] + (["Id_Number"] if "Id_Number" in frame else [])].copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].fillna("").astype(str).str.strip()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("string").str.strip()
    invalid = frame[TEXT_COLUMN].eq("") | frame[TARGET_COLUMN].isna() | frame[TARGET_COLUMN].eq("")
    if invalid.any():
        raise ValueError(f"{split_name} contains {int(invalid.sum())} empty text or target rows")
    if "Id_Number" in frame and frame["Id_Number"].duplicated().any():
        raise ValueError(f"{split_name} contains duplicate Id_Number values")
    return frame


def make_pipeline(c_value: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=150_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LinearSVC(C=c_value, class_weight="balanced", random_state=RANDOM_SEED),
            ),
        ]
    )


def score_model(model: Pipeline, frame: pd.DataFrame, labels: list[str]) -> dict[str, float]:
    truth = frame[TARGET_COLUMN]
    predicted = model.predict(frame[TEXT_COLUMN])
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(precision_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    train = load_split(args.data_dir / "train.csv", "train")
    validation = load_split(args.data_dir / "val.csv", "validation")
    test = load_split(args.data_dir / "test.csv", "test")

    if all("Id_Number" in frame for frame in (train, validation, test)):
        id_sets = [set(frame["Id_Number"]) for frame in (train, validation, test)]
        if (id_sets[0] & id_sets[1]) or (id_sets[0] & id_sets[2]) or (id_sets[1] & id_sets[2]):
            raise ValueError("Train, validation, and test contain overlapping Id_Number values")

    labels = sorted(set(train[TARGET_COLUMN]) | set(validation[TARGET_COLUMN]) | set(test[TARGET_COLUMN]))
    unseen = set(validation[TARGET_COLUMN]) - set(train[TARGET_COLUMN])
    unseen |= set(test[TARGET_COLUMN]) - set(train[TARGET_COLUMN])
    if unseen:
        raise ValueError(f"Labels absent from training data: {sorted(unseen)}")

    selection: list[dict[str, Any]] = []
    best_c: float | None = None
    best_f1 = -1.0
    for c_value in C_VALUES:
        candidate = make_pipeline(c_value)
        candidate.fit(train[TEXT_COLUMN], train[TARGET_COLUMN])
        metrics = score_model(candidate, validation, labels)
        selection.append({"C": c_value, **metrics})
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_c = c_value

    assert best_c is not None
    # Refit after model selection, using all non-test examples.
    development = pd.concat([train, validation], ignore_index=True)
    final_model = make_pipeline(best_c)
    final_model.fit(development[TEXT_COLUMN], development[TARGET_COLUMN])

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "dominant_distortion_tfidf_linearsvc.joblib"
    joblib.dump(
        {
            "pipeline": final_model,
            "text_column": TEXT_COLUMN,
            "target_column": TARGET_COLUMN,
            "labels": labels,
            "selected_C": best_c,
            "random_seed": RANDOM_SEED,
            "feature_config": "word (1,2) + char_wb (3,5) TF-IDF; balanced LinearSVC",
        },
        model_path,
    )

    truth = test[TARGET_COLUMN]
    predicted = final_model.predict(test[TEXT_COLUMN])
    test_metrics = score_model(final_model, test, labels)
    report = classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("actual").to_csv(
        args.results_dir / "baseline_confusion_matrix.csv"
    )

    metrics_payload = {
        "seed": RANDOM_SEED,
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selected_C": best_c,
        "validation_selection": selection,
        "test_metrics": test_metrics,
        "test_classification_report": report,
        "model_path": str(model_path),
        "classes": labels,
    }
    (args.results_dir / "baseline_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_lines = [
        "# Dominant Distortion Baseline: Model Development Report",
        "",
        "## Method",
        "",
        "Input: `Patient Question`; target: `Dominant Distortion` (11 classes).",
        "Features: word TF-IDF (unigrams and bigrams) plus character-boundary TF-IDF (3–5 grams).",
        "Classifier: class-weighted LinearSVC. The existing stratified 80/10/10 splits were used without modification.",
        f"Regularization C was selected from `{', '.join(map(str, C_VALUES))}` using validation macro-F1; the final model was refit on train + validation (seed {RANDOM_SEED}) and evaluated once on test.",
        "",
        "## Split sizes",
        "",
        f"- Train: {len(train)}",
        f"- Validation: {len(validation)}",
        f"- Test: {len(test)}",
        "",
        "## Validation model selection",
        "",
        "| C | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selection:
        report_lines.append(
            f"| {row['C']} | {row['accuracy']:.4f} | {row['macro_precision']:.4f} | {row['macro_recall']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} |"
        )
    report_lines += [
        "",
        "## Held-out test metrics",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    report_lines.extend(f"| {key.replace('_', ' ').title()} | {value:.4f} |" for key, value in test_metrics.items())
    report_lines += [
        "",
        "Per-class precision, recall, and F1 are in `baseline_metrics.json`; the confusion matrix is `baseline_confusion_matrix.csv`.",
        "",
        f"Saved model: `{model_path.relative_to(PROJECT_ROOT).as_posix()}`.",
        "",
        "This is a research baseline, not a clinical assessment tool. Review per-class errors and validate on representative, independently collected data before any user-facing use.",
        "",
    ]
    (args.results_dir / "model_development_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Selected C={best_c} by validation macro-F1={best_f1:.4f}")
    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved report: {args.results_dir / 'model_development_report.md'}")


if __name__ == "__main__":
    main()
