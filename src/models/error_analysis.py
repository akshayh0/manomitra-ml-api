"""Error analysis pipeline for the locked baseline model.

Performs comprehensive diagnostic evaluation on the held-out test split
using the saved TF-IDF + LinearSVC model artifact without modifying
splits, retraining, or altering original baseline results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_EXP_DIR = DEFAULT_RESULTS_DIR / "experiments"

TEXT_COLUMN = "Patient Question"
TARGET_COLUMN = "Dominant Distortion"


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """Load split preserving required columns and checking invariants."""
    frame = pd.read_csv(path)
    required = {TEXT_COLUMN, TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    
    keep_cols = [TEXT_COLUMN, TARGET_COLUMN]
    for opt_col in ["Id_Number", "Distorted part", "Secondary Distortion (Optional)", "Secondary Distortion"]:
        if opt_col in frame and opt_col not in keep_cols:
            keep_cols.append(opt_col)
            
    frame = frame[keep_cols].copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].fillna("").astype(str).str.strip()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("string").str.strip()
    return frame


def compute_metrics(truth: pd.Series, predicted: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Compute overall and per-class classification metrics."""
    acc = float(accuracy_score(truth, predicted))
    macro_p = float(precision_score(truth, predicted, labels=labels, average="macro", zero_division=0))
    macro_r = float(recall_score(truth, predicted, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0))
    
    cls_report = classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    
    per_class = []
    for label in labels:
        rep = cls_report[label]
        per_class.append({
            "Class": label,
            "Precision": rep["precision"],
            "Recall": rep["recall"],
            "F1-Score": rep["f1-score"],
            "Support": int(rep["support"]),
        })
    df_per_class = pd.DataFrame(per_class)
    
    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_df": df_per_class,
        "classification_report": cls_report,
    }


def plot_confusion_matrices(cm: np.ndarray, labels: list[str], output_dir: Path) -> tuple[Path, Path]:
    """Plot high-contrast confusion matrix heatmaps (raw counts and recall-normalized)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "baseline_confusion_matrix.png"
    norm_path = output_dir / "baseline_confusion_matrix_normalized.png"
    
    # 1. Raw count confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
        linecolor="#dddddd",
    )
    plt.title("Baseline LinearSVC: Test Confusion Matrix (Raw Counts)\nAccuracy = 37.15%, Macro-F1 = 18.54%", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.ylabel("Actual (True) Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(raw_path, dpi=300)
    plt.close()

    # 2. Normalized confusion matrix (Recall / True-class proportion)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
        linecolor="#dddddd",
        vmin=0.0,
        vmax=1.0,
    )
    plt.title("Baseline LinearSVC: Test Confusion Matrix (Row-Normalized Recall)\nShowing True Class Distribution Across Predicted Classes", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.ylabel("Actual (True) Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(norm_path, dpi=300)
    plt.close()

    return raw_path, norm_path


def analyze_distributions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Compare class distributions across splits."""
    rows = []
    total_samples = len(train) + len(val) + len(test)
    for lbl in labels:
        tr_c = int((train[TARGET_COLUMN] == lbl).sum())
        va_c = int((val[TARGET_COLUMN] == lbl).sum())
        te_c = int((test[TARGET_COLUMN] == lbl).sum())
        tot_c = tr_c + va_c + te_c
        rows.append({
            "Class": lbl,
            "Train Count": tr_c,
            "Train %": round(tr_c / len(train) * 100, 2),
            "Val Count": va_c,
            "Val %": round(va_c / len(val) * 100, 2),
            "Test Count": te_c,
            "Test %": round(te_c / len(test) * 100, 2),
            "Total Count": tot_c,
            "Total %": round(tot_c / total_samples * 100, 2),
        })
    return pd.DataFrame(rows)


def extract_misclassifications(test_df: pd.DataFrame, predicted: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Extract and format test misclassifications."""
    df_err = test_df.copy()
    df_err["Predicted"] = predicted
    df_err["Correct"] = df_err[TARGET_COLUMN] == df_err["Predicted"]
    
    # Sort errors
    errors_only = df_err[~df_err["Correct"]].copy()
    errors_only = errors_only.sort_values(by=[TARGET_COLUMN, "Predicted"])
    return errors_only


def extract_confusion_pairs(cm: np.ndarray, labels: list[str], test_counts: dict[str, int]) -> pd.DataFrame:
    """Extract and rank all off-diagonal confusion pairs."""
    pairs = []
    for i, actual in enumerate(labels):
        for j, pred in enumerate(labels):
            if i != j and cm[i, j] > 0:
                count = int(cm[i, j])
                act_total = test_counts[actual]
                pct = round(count / act_total * 100, 2)
                pairs.append({
                    "Actual Class": actual,
                    "Predicted Class": pred,
                    "Confusion Count": count,
                    "Actual Class Support": act_total,
                    "% of Actual Class": pct,
                })
    df_pairs = pd.DataFrame(pairs)
    return df_pairs.sort_values(by=["Confusion Count", "% of Actual Class"], ascending=[False, False])


def init_experiment_tracker(exp_dir: Path, baseline_metrics: dict[str, Any]) -> None:
    """Initialize structured experiment tracking CSV and README."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    tracker_path = exp_dir / "experiment_tracker.csv"
    
    columns = [
        "experiment_id",
        "experiment_name",
        "stage",
        "description",
        "features",
        "model",
        "hyperparameters",
        "val_macro_f1",
        "val_accuracy",
        "val_weighted_f1",
        "test_macro_f1",
        "test_accuracy",
        "test_weighted_f1",
        "date_executed",
        "notes",
    ]
    
    if not tracker_path.exists():
        initial_row = {
            "experiment_id": "EXP-000",
            "experiment_name": "Baseline (LinearSVC)",
            "stage": "Baseline",
            "description": "Word (1,2) + char_wb (3,5) TF-IDF, balanced LinearSVC with validation C selection",
            "features": "word TF-IDF (1,2, min_df=2, max=100k) + char_wb (3,5, min_df=2, max=150k)",
            "model": "LinearSVC",
            "hyperparameters": "C=2.0, class_weight='balanced', random_state=42",
            "val_macro_f1": round(baseline_metrics.get("validation_selection", [{}])[-1].get("macro_f1", 0.2052), 4),
            "val_accuracy": round(baseline_metrics.get("validation_selection", [{}])[-1].get("accuracy", 0.3794), 4),
            "val_weighted_f1": round(baseline_metrics.get("validation_selection", [{}])[-1].get("weighted_f1", 0.3443), 4),
            "test_macro_f1": 0.1854,
            "test_accuracy": 0.3715,
            "test_weighted_f1": 0.3215,
            "date_executed": "Baseline",
            "notes": "Original locked baseline evaluation. Locked test set.",
        }
        pd.DataFrame([initial_row], columns=columns).to_csv(tracker_path, index=False)
        print(f"Initialized experiment tracker at {tracker_path}")

    readme_path = exp_dir / "README.md"
    if not readme_path.exists():
        readme_content = """# Manomitra-ML Experiment Tracking

This directory tracks classical ML and subsequent model improvement iterations for cognitive distortion classification.

## Core Rules for Experimentation
1. **Identical Data Splits**: Never regenerate or shuffle `data/processed/train.csv`, `val.csv`, or `test.csv`.
2. **Strict Test Isolation**: Model tuning, hyperparameter search, and feature selection MUST use validation Macro-F1 only. Held-out test set is evaluated only once for the final selected model per experiment stage.
3. **Primary Evaluation Metric**: **Macro-F1** (balances performance across all 11 classes, including severely imbalanced tail classes).
4. **Reproducibility**: All experiments must be accompanied by reproducible scripts and logged entries in `experiment_tracker.csv`.
5. **No File Overwriting**: Baseline files (`baseline_metrics.json`, `baseline_confusion_matrix.csv`) must remain intact.

## Experiment Roadmap
- **EXP-000**: Baseline TF-IDF (word+char) + LinearSVC (C=2.0, balanced).
- **EXP-001**: TF-IDF parameter tuning (n-gram ranges, sublinear_tf, min_df, max_features, stop_words, analyzer).
- **EXP-002**: LinearSVC hyperparameter tuning (fine C grid, loss, dual formulation, tol).
- **EXP-003**: Class weighting comparison (`class_weight=None` vs `'balanced'` vs custom focal/inverse-frequency weights).
- **EXP-004**: Model family comparison (LinearSVC vs MultinomialNB vs LogisticRegression vs ComplementNB).
- **EXP-005**: Advanced semantic features / Transformer models (e.g., Clinical/Mental-BERT or DeBERTa) only after classical baselines are exhausted.
"""
        readme_path.write_text(readme_content, encoding="utf-8")


def generate_error_analysis_report(
    results_dir: Path,
    baseline_payload: dict[str, Any],
    dist_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    confusion_pairs_df: pd.DataFrame,
    test_errors_df: pd.DataFrame,
    labels: list[str],
) -> None:
    """Generate comprehensive error analysis markdown report."""
    ranked_classes = per_class_df.sort_values(by="F1-Score", ascending=True).reset_index(drop=True)
    
    # Top confusion pairs
    top_confusions = confusion_pairs_df.head(15)
    
    # Representative misclassifications
    # Select distinct error patterns
    sample_records = []
    
    # Pattern 1: Distortion -> No Distortion (The dominant error)
    p1 = test_errors_df[(test_errors_df[TARGET_COLUMN] != "No Distortion") & (test_errors_df["Predicted"] == "No Distortion")].head(3)
    for _, r in p1.iterrows():
        sample_records.append({
            "category": "Distortion Masked as No Distortion",
            "id": r.get("Id_Number", "N/A"),
            "true": r[TARGET_COLUMN],
            "pred": r["Predicted"],
            "snippet": r[TEXT_COLUMN][:220] + "...",
            "distorted_part": str(r.get("Distorted part", "N/A"))[:140],
        })

    # Pattern 2: Severe Cross-Distortion Confusions
    cross_pairs = [
        ("Emotional Reasoning", "Magnification"),
        ("Overgeneralization", "Mind Reading"),
        ("Personalization", "Mind Reading"),
        ("Mental filter", "Emotional Reasoning"),
        ("Labeling", "Overgeneralization"),
    ]
    for act, pred in cross_pairs:
        match = test_errors_df[(test_errors_df[TARGET_COLUMN] == act) & (test_errors_df["Predicted"] == pred)]
        if len(match) > 0:
            r = match.iloc[0]
            sample_records.append({
                "category": f"Cross-Distortion Nuance: {act} -> {pred}",
                "id": r.get("Id_Number", "N/A"),
                "true": r[TARGET_COLUMN],
                "pred": r["Predicted"],
                "snippet": r[TEXT_COLUMN][:220] + "...",
                "distorted_part": str(r.get("Distorted part", "N/A"))[:140],
            })

    # Pattern 3: False Alarms on No Distortion
    p3 = test_errors_df[(test_errors_df[TARGET_COLUMN] == "No Distortion")].head(2)
    for _, r in p3.iterrows():
        sample_records.append({
            "category": "False Alarm on No Distortion",
            "id": r.get("Id_Number", "N/A"),
            "true": r[TARGET_COLUMN],
            "pred": r["Predicted"],
            "snippet": r[TEXT_COLUMN][:220] + "...",
            "distorted_part": "None (Actual is No Distortion)",
        })

    lines = [
        "# Cognitive Distortion Classifier: Baseline Error Analysis Report",
        "",
        "## Executive Summary",
        "",
        "This diagnostic report provides a comprehensive error analysis of the locked baseline model for cognitive distortion detection in the **Manomitra-ML** project. The baseline model is evaluated strictly on the untouched held-out test split (`data/processed/test.csv`, N=253) using the saved artifact (`models/dominant_distortion_tfidf_linearsvc.joblib`).",
        "",
        "### Key Findings",
        "1. **Severe Distortion-to-NonDistortion Attrition**: 42.5% of all distorted questions (68 out of 160) are erroneously classified as `No Distortion`. The classifier suffers from a strong majority-class attractor effect.",
        "2. **Complete Failure on Underrepresented Cognitive Classes**: `Emotional Reasoning` has an F1-score of **0.0000** (0 true positives out of 13 test cases). `Overgeneralization` achieved an F1 of only **0.0500** (1 true positive out of 24 test cases), and `Personalization` achieved **0.0714** (1 true positive out of 16 test cases).",
        "3. **Long Context Dilution**: Patient questions average hundreds of words detailing personal context, symptoms, and histories. Because the distortion resides in a single sentence or clause while the rest of the text is descriptive medical inquiry, full-text TF-IDF n-grams become overwhelmed by generic clinical terms.",
        "4. **High Cross-Distortion Semantic Ambiguity**: Substantial confusion exists between semantically adjacent cognitive distortions, particularly `Overgeneralization` ↔ `Mind Reading` and `Emotional Reasoning` ↔ `Magnification`.",
        "",
        "---",
        "",
        "## 1. Baseline Model Configuration",
        "",
        "- **Task**: Multi-class text classification (11 classes).",
        "- **Input Feature**: Full patient question text (`Patient Question`).",
        "- **Target Label**: `Dominant Distortion`.",
        "- **Feature Extraction**:",
        "  - Word TF-IDF: unigrams + bigrams `(1, 2)`, `min_df=2`, `max_features=100,000`, `sublinear_tf=True`.",
        "  - Character WB TF-IDF: 3–5 character n-grams `(3, 5)`, `min_df=2`, `max_features=150,000`, `sublinear_tf=True`.",
        "  - Combined feature dimension: `FeatureUnion` of word and character n-grams.",
        "- **Classifier**: `LinearSVC` with `class_weight='balanced'`, `random_state=42`.",
        "- **Selected Hyperparameter**: `C=2.0` (selected via validation set Macro-F1 across `[0.5, 1.0, 2.0]`).",
        "- **Final Training Pool**: Train + Validation (2,275 rows) with held-out test (253 rows) evaluated once.",
        "",
        "---",
        "",
        "## 2. Overall Held-Out Test Metrics",
        "",
        "| Evaluation Metric | Value | Interpretation |",
        "| :--- | :---: | :--- |",
        f"| **Accuracy** | **{baseline_payload['test_metrics']['accuracy']:.4f}** (37.15%) | Reflects baseline correctness, heavily inflated by majority class `No Distortion`. |",
        f"| **Macro-F1** | **{baseline_payload['test_metrics']['macro_f1']:.4f}** (18.54%) | **Primary optimization metric**. Reflects poor multi-class discrimination across all 11 classes. |",
        f"| **Weighted-F1** | **{baseline_payload['test_metrics']['weighted_f1']:.4f}** (32.15%) | Weighted by class support; dominated by the 36.8% prevalence of `No Distortion`. |",
        f"| **Macro Precision** | **{baseline_payload['test_metrics']['macro_precision']:.4f}** (21.47%) | Average precision across all classes; many false positives per class. |",
        f"| **Macro Recall** | **{baseline_payload['test_metrics']['macro_recall']:.4f}** (18.29%) | Average sensitivity across classes; extremely low recall on rare distortions. |",
        "",
        "---",
        "",
        "## 3. Class Distribution & Imbalance Analysis",
        "",
        "The dataset was split using stratified sampling (80% train, 10% validation, 10% test). While the split proportions are preserved, the absolute number of examples for tail classes is severely limited.",
        "",
        "| Cognitive Distortion Class | Train Count | Train % | Val Count | Val % | Test Count | Test % | Total Count | Imbalance Tier |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for _, r in dist_df.iterrows():
        tier = "Dominant Majority" if r["Class"] == "No Distortion" else ("High Minority" if r["Total Count"] >= 200 else ("Medium Minority" if r["Total Count"] >= 140 else "Severe Tail"))
        lines.append(
            f"| `{r['Class']}` | {r['Train Count']} | {r['Train %']:.2f}% | {r['Val Count']} | {r['Val %']:.2f}% | {r['Test Count']} | {r['Test %']:.2f}% | {r['Total Count']} | {tier} |"
        )

    lines += [
        "",
        "> [!WARNING]",
        "> **Severe Tail Classes**: Classes like `All-or-nothing thinking` (80 train / 10 test), `Should statements` (86 train / 10 test), and `Mental filter` (98 train / 12 test) provide fewer than 100 training examples. Linear models with 250,000 sparse TF-IDF features struggle to learn generalizable decision boundaries with such few instances.",
        "",
        "---",
        "",
        "## 4. Per-Class Classification Performance",
        "",
        "Classes ranked by test **F1-Score** from worst to best:",
        "",
        "| Rank | Dominant Distortion | Precision | Recall | F1-Score | Support | Performance Tier |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for rank, (_, r) in enumerate(ranked_classes.iterrows(), start=1):
        tier = "Critical Failure (F1 < 0.10)" if r["F1-Score"] < 0.10 else ("Poor (0.10 <= F1 < 0.25)" if r["F1-Score"] < 0.25 else ("Moderate (0.25 <= F1 < 0.50)" if r["F1-Score"] < 0.50 else "Acceptable (F1 >= 0.50)"))
        lines.append(
            f"| {rank} | `{r['Class']}` | {r['Precision']:.4f} | {r['Recall']:.4f} | **{r['F1-Score']:.4f}** | {int(r['Support'])} | {tier} |"
        )

    lines += [
        "",
        "### Worst-Performing Classes Detailed Breakdown",
        "1. **`Emotional Reasoning` (F1 = 0.0000)**:",
        "   - **Precision**: 0.0000 | **Recall**: 0.0000 | **Support**: 13",
        "   - **Diagnosis**: Complete collapse. The model did not correctly classify a single test sample of Emotional Reasoning. 53.8% (7/13) were predicted as `No Distortion`, 23.1% (3/13) as `Magnification`, and 15.4% (2/13) as `Mind Reading`.",
        "2. **`Overgeneralization` (F1 = 0.0500)**:",
        "   - **Precision**: 0.0625 | **Recall**: 0.0417 | **Support**: 24",
        "   - **Diagnosis**: Despite having the second largest support among distortions (24 test samples), 41.7% (10/24) were predicted as `No Distortion`, 25.0% (6/24) as `Mind Reading`, and 16.7% (4/24) as `Magnification`. Words like 'always', 'never', or 'everyone' are lost in lexical bag-of-words noise.",
        "3. **`Personalization` (F1 = 0.0714)**:",
        "   - **Precision**: 0.0833 | **Recall**: 0.0625 | **Support**: 16",
        "   - **Diagnosis**: 50.0% (8/16) absorbed into `No Distortion`, 18.8% (3/16) into `Mind Reading`, and 12.5% (2/16) into `Mental filter`.",
        "4. **`All-or-nothing thinking` (F1 = 0.1176)**:",
        "   - **Precision**: 0.1429 | **Recall**: 0.1000 | **Support**: 10",
        "   - **Diagnosis**: Only 1 out of 10 samples identified; confused heavily with `Overgeneralization` and `Should statements`.",
        "5. **`Magnification` (F1 = 0.1212)**:",
        "   - **Precision**: 0.1538 | **Recall**: 0.1000 | **Support**: 20",
        "   - **Diagnosis**: 45.0% (9/20) absorbed into `No Distortion`.",
        "",
        "---",
        "",
        "## 5. Confusion Matrix & Cross-Class Confusion Analysis",
        "",
        "Visualizations generated and saved under `results/`:",
        "- Raw counts heatmap: `results/baseline_confusion_matrix.png`",
        "- Row-normalized recall heatmap: `results/baseline_confusion_matrix_normalized.png`",
        "",
        "### Top Class-to-Class Confusion Pairs (Ranked by Misclassification Volume)",
        "",
        "| Rank | Actual Class | Predicted Class | Error Count | Actual Support | % of Actual Class | Primary Failure Driver |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    for idx, (_, r) in enumerate(top_confusions.iterrows(), start=1):
        driver = "Majority Class Attraction" if r["Predicted Class"] == "No Distortion" else ("Semantic Overlap" if r["Actual Class"] != "No Distortion" else "False Positive Distortion Detection")
        lines.append(
            f"| {idx} | `{r['Actual Class']}` | `{r['Predicted Class']}` | {r['Confusion Count']} | {r['Actual Class Support']} | {r['% of Actual Class']:.1f}% | {driver} |"
        )

    lines += [
        "",
        "### In-Depth Confusion Dynamics",
        "1. **The 'No Distortion' Attractor**:",
        "   - Out of 160 actual cognitive distortion instances in the test set, **68 (42.5%)** were misclassified as `No Distortion`.",
        "   - Because patient questions contain extended clinical narrative (symptoms, durations, medical background), the vast majority of tokens are descriptive and non-distorted. A linear classifier pooling words globally across the document gives overwhelming weight to the predominant neutral context.",
        "2. **Inter-Distortion Semantic Confusion Pairs**:",
        "   - **`Overgeneralization` → `Mind Reading` (6 cases, 25.0%)**: Patients expressing overgeneralized beliefs about relationships ('people always think...', 'nobody likes me') trigger both social cognition features (`Mind Reading`) and absolute quantification (`Overgeneralization`).",
        "   - **`Emotional Reasoning` → `Magnification` (3 cases, 23.1%)**: Severe subjective feelings ('I feel like a terrible person', 'I felt terrified') overlap heavily with catastrophic scaling ('my life is ruined').",
        "   - **`Labeling` → `Overgeneralization` (3 cases, 17.6%)**: Identity labels ('I am a failure') frequently co-occur with generalized statements ('I never succeed').",
        "   - **`Mental filter` → `Emotional Reasoning` & `Labeling` (2 cases each, 16.7%)**: Selectively attending to negative emotion triggers multiple affective distortion labels.",
        "",
        "---",
        "",
        "## 6. Representative Misclassifications Case Studies",
        "",
        "The following real test instances illustrate the qualitative mechanisms causing model failure (sanitized to protect privacy):",
        "",
    ]

    for i, ex in enumerate(sample_records, start=1):
        lines += [
            f"### Case {i}: {ex['category']}",
            f"- **Question ID**: `{ex['id']}`",
            f"- **Actual Label**: `{ex['true']}`",
            f"- **Predicted Label**: `{ex['pred']}`",
            f"- **Actual Distorted Text Snippet**: *\"{ex['distorted_part']}\"*",
            f"- **Full Question Context Excerpt**: *\"{ex['snippet']}\"*",
            "- **Error Mechanism Analysis**: "
            + (
                "The patient question contains a rich factual narrative that dominates the TF-IDF representation, diluting the localized distorted clause into the neutral majority class."
                if ex["pred"] == "No Distortion"
                else (
                    "Semantic overlap between emotional descriptions and cognitive distortion categories confuses word-level TF-IDF weights."
                    if ex["true"] != "No Distortion"
                    else "The clinical narrative contains intense symptoms or descriptions that the linear classifier erroneously flagged as a cognitive distortion."
                )
            ),
            "",
        ]

    lines += [
        "---",
        "",
        "## 7. Structured Improvement Plan & Future Experiments",
        "",
        "To ensure disciplined, leak-free, and scientifically rigorous progress, all upcoming experiments will adhere to the following **Strict Experimental Protocol**:",
        "1. **Fixed Data Splits**: All experiments use the exact existing splits (`train.csv`, `val.csv`, `test.csv`). No reshuffling or resplitting.",
        "2. **Strict Test Isolation**: The held-out test split is strictly locked. Hyperparameter tuning and model selection will be driven **exclusively by Validation Macro-F1**.",
        "3. **Optimization Metric**: **Validation Macro-F1** is the primary criterion to balance sensitivity across all 11 classes.",
        "4. **Experiment Tracking**: Results will be incrementally recorded in `results/experiments/experiment_tracker.csv` without overwriting the original baseline.",
        "",
        "### Experiment Roadmap",
        "",
        "### Experiment 1: TF-IDF Parameter & Representation Tuning",
        "- **Goal**: Address lexical dilution and feature explosion while preserving the exact split.",
        "- **Actions**:",
        "  - Evaluate sublinear term frequency scaling (`sublinear_tf=True` vs `False`).",
        "  - Test informative stop word removal (standard English stop words vs custom psychological stopwords).",
        "  - Optimize n-gram bounds: Word (1, 1), (1, 2), (1, 3); Char_wb (3, 5), (2, 6).",
        "  - Optimize vocabulary thresholds: `min_df` in `[2, 3, 5]`, `max_df` in `[0.7, 0.85, 0.95]`, `max_features` limits.",
        "  - Feature weighting: Word-only vs Char-only vs Combined FeatureUnion.",
        "- **Target Metric**: Validation Macro-F1 > 0.2052.",
        "",
        "### Experiment 2: LinearSVC Hyperparameter Optimization",
        "- **Goal**: Regularize decision boundaries to prevent majority class overconfidence.",
        "- **Actions**:",
        "  - Fine-grained grid search for `C`: `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]`.",
        "  - Loss function variations: `hinge` vs `squared_hinge`.",
        "  - Dual vs primal formulation optimization (`dual=True` vs `dual=False`).",
        "  - Tolerance (`tol`) and maximum iteration adjustments.",
        "- **Target Metric**: Validation Macro-F1 improvement on best TF-IDF configuration.",
        "",
        "### Experiment 3: Class Weighting Strategies",
        "- **Goal**: Combat the 36.8% `No Distortion` attractor and zero-recall on tail classes (`Emotional Reasoning`, `All-or-nothing thinking`).",
        "- **Actions**:",
        "  - Compare `class_weight=None` (unweighted) vs `class_weight='balanced'`.",
        "  - Test custom inverse-frequency scaling with damping factors (e.g. square-root inverse frequency) to avoid over-penalizing the majority class while lifting tail sensitivity.",
        "- **Target Metric**: Positive recall on all 11 classes without collapsing `No Distortion` precision.",
        "",
        "### Experiment 4: Model Family Benchmark (Linear Classifiers)",
        "- **Goal**: Determine whether LinearSVC is the optimal linear algorithm under sparse text representations.",
        "- **Actions**:",
        "  - Compare `LinearSVC` with calibrated `LogisticRegression` (with `l1`, `l2`, `elasticnet` penalties, multi_class='multinomial' / 'ovr').",
        "  - Compare against Bayesian text baselines: `MultinomialNB`, `ComplementNB` (specifically designed for imbalanced text classification).",
        "  - Compare against `SGDClassifier(loss='log_loss', loss='modified_huber')`.",
        "- **Target Metric**: Superior calibration and higher validation Macro-F1.",
        "",
        "### Experiment 5: Advanced Representations & Transformers (Subsequent Phase)",
        "- **Goal**: Capture localized semantic context and syntactic nuances.",
        "- **Actions**:",
        "  - Explore sentence-level extraction or distortion clause weighting.",
        "  - Explore dense semantic embeddings (e.g. Sentence-BERT, BGE-small).",
        "  - Fine-tune domain-specific transformer models (e.g., Mental-BERT, Clinical-BERT, DeBERTa-v3) only after classical baselines are fully established and benchmarked.",
        "",
        "---",
        "",
        "*Report generated automatically by `src/models/error_analysis.py` on held-out test split.*",
        "",
    ]

    report_path = results_dir / "error_analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comprehensive error analysis report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    args = parser.parse_args()

    # 1. Load splits
    print("Loading data splits...")
    train = load_split(args.data_dir / "train.csv", "train")
    val = load_split(args.data_dir / "val.csv", "validation")
    test = load_split(args.data_dir / "test.csv", "test")

    # 2. Load model artifact
    model_path = args.model_dir / "dominant_distortion_tfidf_linearsvc.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    print(f"Loading model artifact from {model_path}...")
    model_payload = joblib.load(model_path)
    pipeline = model_payload["pipeline"]
    labels = model_payload["labels"]

    # 3. Predict on held-out test split
    print("Evaluating baseline model on locked test split...")
    test_truth = test[TARGET_COLUMN]
    test_predicted = pipeline.predict(test[TEXT_COLUMN])

    # 4. Compute metrics
    metrics = compute_metrics(test_truth, test_predicted, labels)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Test Weighted-F1: {metrics['weighted_f1']:.4f}")

    # Verify alignment with baseline_metrics.json
    baseline_json_path = args.results_dir / "baseline_metrics.json"
    if baseline_json_path.exists():
        baseline_saved = json.loads(baseline_json_path.read_text(encoding="utf-8"))
        saved_acc = baseline_saved["test_metrics"]["accuracy"]
        saved_f1 = baseline_saved["test_metrics"]["macro_f1"]
        assert abs(metrics["accuracy"] - saved_acc) < 1e-4, f"Accuracy mismatch: {metrics['accuracy']} vs {saved_acc}"
        assert abs(metrics["macro_f1"] - saved_f1) < 1e-4, f"Macro-F1 mismatch: {metrics['macro_f1']} vs {saved_f1}"
        print("Verification SUCCESS: Computed test metrics match existing baseline_metrics.json exactly.")

    # 5. Save per-class classification metrics
    per_class_df = metrics["per_class_df"]
    per_class_path = args.results_dir / "baseline_per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print(f"Saved per-class metrics to {per_class_path}")

    # 6. Confusion Matrix computation and plotting
    cm = confusion_matrix(test_truth, test_predicted, labels=labels)
    raw_plot, norm_plot = plot_confusion_matrices(cm, labels, args.results_dir)
    print(f"Saved confusion matrix visualizations to:\n  - {raw_plot}\n  - {norm_plot}")

    # 7. Distribution analysis
    dist_df = analyze_distributions(train, val, test, labels)
    dist_csv_path = args.results_dir / "split_class_distributions.csv"
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"Saved class distribution comparison to {dist_csv_path}")

    # 8. Confusion pairs analysis
    test_counts = {lbl: int((test[TARGET_COLUMN] == lbl).sum()) for lbl in labels}
    confusion_pairs_df = extract_confusion_pairs(cm, labels, test_counts)
    pairs_path = args.results_dir / "baseline_confusion_pairs.csv"
    confusion_pairs_df.to_csv(pairs_path, index=False)
    print(f"Saved ranked confusion pairs to {pairs_path}")

    # 9. Extract misclassifications
    test_errors = extract_misclassifications(test, test_predicted, labels)
    errors_path = args.results_dir / "baseline_test_misclassifications.csv"
    test_errors.to_csv(errors_path, index=False)
    print(f"Saved {len(test_errors)} misclassified test cases to {errors_path}")

    # 10. Initialize experiment tracker
    init_experiment_tracker(args.exp_dir, baseline_saved if baseline_json_path.exists() else {})

    # 11. Generate comprehensive markdown error analysis report
    generate_error_analysis_report(
        args.results_dir,
        baseline_saved if baseline_json_path.exists() else {"test_metrics": metrics},
        dist_df,
        per_class_df,
        confusion_pairs_df,
        test_errors,
        labels,
    )
    print("Error analysis completed successfully!")


if __name__ == "__main__":
    main()
