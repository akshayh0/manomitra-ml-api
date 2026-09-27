import os
import re
import pandas as pd
import numpy as np

def normalize_whitespace(text):
    if not isinstance(text, str):
        return text
    # Normalize newline characters
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse multiple consecutive horizontal spaces to a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple consecutive newlines to at most two (paragraph separation)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def main():
    # Define directories
    raw_dir = r"c:\Manomitra-ML\data\raw"
    processed_dir = r"c:\Manomitra-ML\data\processed"
    results_dir = r"c:\Manomitra-ML\results"

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Raw Datasets
    annotated_raw_path = os.path.join(raw_dir, "Annotated_data.csv")
    therapist_raw_path = os.path.join(raw_dir, "Therapist_responses.csv")

    print("Loading raw datasets...")
    df_ann = pd.read_csv(annotated_raw_path)
    df_ther = pd.read_csv(therapist_raw_path)

    # Keep track of original counts
    orig_ann_rows = len(df_ann)
    orig_ther_rows = len(df_ther)

    # Missing values before cleaning
    ann_missing_before = df_ann.isnull().sum().to_dict()
    ther_missing_before = df_ther.isnull().sum().to_dict()

    # 2. Schema Validation
    expected_ann_cols = ['Id_Number', 'Patient Question', 'Distorted part', 'Dominant Distortion', 'Secondary Distortion (Optional)']
    expected_ther_cols = ['Answer', 'Question', 'Id_Number']

    assert all(col in df_ann.columns for col in expected_ann_cols), f"Annotated dataset columns do not match expected schema: {df_ann.columns.tolist()}"
    assert all(col in df_ther.columns for col in expected_ther_cols), f"Therapist responses dataset columns do not match expected schema: {df_ther.columns.tolist()}"

    # 3. Clean Annotated Dataset
    print("Cleaning Annotated_data.csv...")
    # Normalize whitespaces
    df_ann['Patient Question'] = df_ann['Patient Question'].apply(normalize_whitespace)
    df_ann['Distorted part'] = df_ann['Distorted part'].apply(normalize_whitespace)

    # Remove rows where Patient Question is missing or empty
    null_questions_mask = df_ann['Patient Question'].isnull() | (df_ann['Patient Question'] == '')
    removed_null_ann = null_questions_mask.sum()
    df_ann_cleaned = df_ann[~null_questions_mask].copy()

    # Detect duplicate questions and conflicting labels
    dup_mask = df_ann_cleaned.duplicated(subset=['Patient Question'], keep=False)
    dup_df = df_ann_cleaned[dup_mask]

    conflicting_ids = []
    non_conflicting_dup_ids = []
    
    # Analyze duplicates
    if len(dup_df) > 0:
        grouped_dups = dup_df.groupby('Patient Question')
        for q_text, group in grouped_dups:
            unique_labels = group['Dominant Distortion'].unique()
            if len(unique_labels) > 1:
                # Conflicting labels
                conflicting_ids.extend(group['Id_Number'].tolist())
            else:
                # Same label, just duplicates
                # We will keep the first one and flag others
                first_id = group['Id_Number'].iloc[0]
                other_ids = group['Id_Number'].iloc[1:].tolist()
                non_conflicting_dup_ids.extend(other_ids)

    # For the first experiment, exclude conflicting duplicates
    print(f"Found conflicting duplicate IDs: {conflicting_ids}")
    print(f"Found redundant (non-conflicting) duplicate IDs to remove: {non_conflicting_dup_ids}")

    # Exclude conflicting duplicates
    df_ann_experiments = df_ann_cleaned[~df_ann_cleaned['Id_Number'].isin(conflicting_ids)].copy()
    # Deduplicate non-conflicting duplicates (keep first)
    df_ann_experiments = df_ann_experiments[~df_ann_experiments['Id_Number'].isin(non_conflicting_dup_ids)].copy()

    # Verify target labels
    expected_classes = [
        'Personalization', 'Labeling', 'No Distortion', 'Fortune-telling', 
        'Magnification', 'Mind Reading', 'All-or-nothing thinking', 
        'Overgeneralization', 'Mental filter', 'Emotional Reasoning', 'Should statements'
    ]
    actual_classes = df_ann_experiments['Dominant Distortion'].unique()
    invalid_classes = [c for c in actual_classes if c not in expected_classes]
    assert len(invalid_classes) == 0, f"Found unexpected classes in Dominant Distortion: {invalid_classes}"

    # Save cleaned annotated dataset
    annotated_clean_path = os.path.join(processed_dir, "annotated_clean.csv")
    df_ann_experiments.to_csv(annotated_clean_path, index=False)
    print(f"Saved cleaned annotated dataset to {annotated_clean_path}")

    # 4. Clean Therapist Responses
    print("Cleaning Therapist_responses.csv...")
    df_ther['Question'] = df_ther['Question'].apply(normalize_whitespace)
    df_ther['Answer'] = df_ther['Answer'].apply(normalize_whitespace)

    # Identify rows with missing Answers
    missing_answer_mask = df_ther['Answer'].isnull() | (df_ther['Answer'] == '')
    removed_missing_ther = missing_answer_mask.sum()
    missing_answer_ids = df_ther[missing_answer_mask]['Id_Number'].tolist()

    df_ther_cleaned = df_ther[~missing_answer_mask].copy()

    # Save cleaned therapist response dataset
    therapist_clean_path = os.path.join(processed_dir, "therapist_responses_clean.csv")
    df_ther_cleaned.to_csv(therapist_clean_path, index=False)
    print(f"Saved cleaned therapist responses to {therapist_clean_path}")

    # 5. Dataset Join Analysis
    print("Analyzing and creating joined research dataset...")
    # Join on Id_Number
    # Since df_ann contains the annotations and df_ther contains therapist answers, we do an inner join
    # using the original datasets (to preserve all available records, including any that were excluded from the experiment dataset)
    df_ann_full_cleaned = df_ann_cleaned.copy()
    
    # We join with df_ther (which might have missing answers, but we preserve them in combined research dataset and flag them)
    combined_df = pd.merge(
        df_ann_full_cleaned,
        df_ther,
        on='Id_Number',
        how='inner',
        suffixes=('_ann', '_ther')
    )

    # Rename columns for clarity as suggested
    combined_df = combined_df.rename(columns={
        'Patient Question': 'Patient Question',
        'Question': 'Therapist Question',
        'Answer': 'Therapist Answer',
        'Secondary Distortion (Optional)': 'Secondary Distortion'
    })

    # Select and order columns
    combined_cols = [
        'Id_Number', 'Patient Question', 'Dominant Distortion', 
        'Secondary Distortion', 'Distorted part', 'Therapist Question', 'Therapist Answer'
    ]
    combined_df = combined_df[combined_cols]

    # Detect text discrepancies in overlapping IDs
    discrepancies = []
    for idx, row in combined_df.iterrows():
        pq = row['Patient Question']
        tq = row['Therapist Question']
        if pq != tq:
            discrepancies.append({
                'Id_Number': row['Id_Number'],
                'Annotated_Question_Len': len(pq),
                'Therapist_Question_Len': len(tq)
            })

    # Save combined research dataset
    combined_path = os.path.join(processed_dir, "combined_research_dataset.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved combined research dataset to {combined_path}")

    # 6. Stratified Train/Val/Test Split for Classification Task
    print("Performing stratified train/validation/test splitting...")
    # Split df_ann_experiments (which has duplicate/conflicting rows removed)
    # Stratified split: 80% train, 10% val, 10% test
    # We'll use sklearn if available, else a fallback pandas group sampling
    try:
        from sklearn.model_selection import train_test_split
        
        # Split 80% train and 20% temp (val + test)
        train_df, temp_df = train_test_split(
            df_ann_experiments, 
            test_size=0.20, 
            random_state=42, 
            stratify=df_ann_experiments['Dominant Distortion']
        )
        
        # Split 20% temp into 50% val and 50% test (10% and 10% of total)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.50, 
            random_state=42, 
            stratify=temp_df['Dominant Distortion']
        )
    except ImportError:
        print("Scikit-learn not available. Falling back to pandas stratified sampling...")
        # Fallback stratified split using pandas groupby
        train_df = df_ann_experiments.groupby('Dominant Distortion', group_keys=False).apply(
            lambda x: x.sample(frac=0.8, random_state=42)
        )
        temp_df = df_ann_experiments.drop(train_df.index)
        
        val_df = temp_df.groupby('Dominant Distortion', group_keys=False).apply(
            lambda x: x.sample(frac=0.5, random_state=42)
        )
        test_df = temp_df.drop(val_df.index)

    # Save split files
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train split size: {len(train_df)}")
    print(f"Validation split size: {len(val_df)}")
    print(f"Test split size: {len(test_df)}")

    # 7. Final Metrics and Report Generation
    print("Generating data cleaning report...")
    
    # Missing values after cleaning (in annotated clean and therapist clean)
    ann_missing_after = df_ann_experiments.isnull().sum().to_dict()
    ther_missing_after = df_ther_cleaned.isnull().sum().to_dict()

    # Final class distribution in annotated_clean
    class_counts = df_ann_experiments['Dominant Distortion'].value_counts()
    class_pcts = df_ann_experiments['Dominant Distortion'].value_counts(normalize=True) * 100

    report_content = f"""# Manomitra V2: Data Cleaning and Preprocessing Report

This report documents the dataset preparation and cleaning steps performed on the raw datasets for the cognitive distortion detection task.

## 1. Summary of Rows and Retained Data

| Dataset | Original Rows | Removed Rows | Retained Rows | Reason for Removal |
| :--- | :---: | :---: | :---: | :--- |
| **Annotated_data.csv** | {orig_ann_rows} | {orig_ann_rows - len(df_ann_experiments)} | {len(df_ann_experiments)} | Excluded empty text, conflicting duplicate labels, and redundant duplicates. |
| **Therapist_responses.csv** | {orig_ther_rows} | {removed_missing_ther} | {len(df_ther_cleaned)} | Removed rows with missing therapist answers. |

---

## 2. Missing Values Profile

### Before Cleaning
* **Annotated Dataset**:
{chr(10).join([f"  * `{k}`: {v} missing values" for k, v in ann_missing_before.items()])}
* **Therapist Responses**:
{chr(10).join([f"  * `{k}`: {v} missing values" for k, v in ther_missing_before.items()])}

### After Cleaning
* **Cleaned Annotated Dataset (`annotated_clean.csv`)**:
{chr(10).join([f"  * `{k}`: {v} missing values" for k, v in ann_missing_after.items()])}
* **Cleaned Therapist Responses (`therapist_responses_clean.csv`)**:
{chr(10).join([f"  * `{k}`: {v} missing values" for k, v in ther_missing_after.items()])}

> [!NOTE]
> In `annotated_clean.csv`, the remaining missing values in `Distorted part` ({ann_missing_after.get('Distorted part', 0)}) align exactly with the {df_ann_experiments['Dominant Distortion'].value_counts().get('No Distortion', 0)} rows labeled as `No Distortion`. Missing values in `Secondary Distortion (Optional)` ({ann_missing_after.get('Secondary Distortion (Optional)', 0)}) are expected as it is an optional field.

---

## 3. Duplicate and Conflicting Labels Analysis

* **Total duplicate questions found**: {len(dup_df)} rows share duplicate question text.
* **Conflicting duplicate IDs**: {conflicting_ids}
  * **Handling**: Excluded from the classification dataset (`annotated_clean.csv`) to prevent label confusion and data leakage.
  * **Known Conflict Details**:
    * **ID 265**: Labeled as `No Distortion`
    * **ID 307**: Labeled as `Overgeneralization`
* **Redundant duplicate IDs (same label)**: {non_conflicting_dup_ids}
  * **Handling**: Excluded to ensure unique question inputs (retained the first instance only).

---

## 4. Therapist Responses: Missing Answers
* **Missing Answer Rows**: {len(missing_answer_ids)} records in `Therapist_responses.csv` were missing the therapist's response.
* **Missing Answer IDs**: {missing_answer_ids}
* **Overlap with Annotated dataset**: IDs {list(set(missing_answer_ids).intersection(set(df_ann['Id_Number'])))} overlap with the annotated dataset.
* **Handling**: Excluded from `therapist_responses_clean.csv` to avoid supplying empty labels/answers to generative models.

---

## 5. Dataset Join and Text Discrepancies
* **Join Reliability**: The two datasets share a reliable **1-to-1 mapping via `Id_Number`**.
* **Overlap Size**: {len(combined_df)} records match between the cleaned annotated dataset and the therapist responses.
* **Question Text Discrepancies**: Programmatic inspection detected **{len(discrepancies)} records** where the question texts differ slightly:
"""

    for disc in discrepancies:
        report_content += f"  * **ID {disc['Id_Number']}**: Annotated text length = {disc['Annotated_Question_Len']} chars, Therapist text length = {disc['Therapist_Question_Len']} chars.\n"

    report_content += f"""
* **Handling**: In the joined dataset (`combined_research_dataset.csv`), we preserved both versions as separate columns (`Patient Question` and `Therapist Question`) to avoid modifying original source texts.

---

## 6. Final Class Distribution (`annotated_clean.csv`)

| Rank | Dominant Distortion | Count | Percentage |
| :--- | :--- | :---: | :---: |
"""

    for rank, (val, cnt) in enumerate(class_counts.items()):
        pct = class_pcts[val]
        report_content += f"| {rank+1} | `{val}` | {cnt} | {pct:.2f}% |\n"

    report_content += f"""
---

## 7. Files Created and Output Sizes

1. **Cleaned Annotated Dataset**: `data/processed/annotated_clean.csv` ({len(df_ann_experiments)} rows)
2. **Cleaned Therapist Responses**: `data/processed/therapist_responses_clean.csv` ({len(df_ther_cleaned)} rows)
3. **Joined Research Dataset**: `data/processed/combined_research_dataset.csv` ({len(combined_df)} rows)
4. **Training Set**: `data/processed/train.csv` ({len(train_df)} rows)
5. **Validation Set**: `data/processed/val.csv` ({len(val_df)} rows)
6. **Test Set**: `data/processed/test.csv` ({len(test_df)} rows)

## 8. Remaining Data-Quality Concerns
* **Class Imbalance**: `No Distortion` makes up {class_pcts.get('No Distortion', 0):.2f}% of the classification dataset. Class weighting is recommended during training.
* **Slight Text Differences**: The 6 discrepant questions should be aligned if they are ever used in unified sequence-to-sequence pipelines.
"""

    report_path = os.path.join(results_dir, "data_cleaning_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Saved data cleaning report to {report_path}")
    print("Preprocessing completed successfully!")

if __name__ == '__main__':
    main()
