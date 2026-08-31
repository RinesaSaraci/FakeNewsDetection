import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

MODEL_DIR = Path("models/bert/binary/final")

ORIGINAL_TEST_PATH = Path(
    "data/raw/liar_dataset/test.tsv"
)

BERT_TEST_DIR = Path(
    "data/processed/bert/binary/test"
)

OUTPUT_DIR = Path(
    "results/bert/original_labels"
)

PLOTS_DIR = OUTPUT_DIR / "plots"

BATCH_SIZE = 8

LABEL_COLUMN = 1
STATEMENT_COLUMN = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LABEL MAPPINGS
# ============================================================

BINARY_MAPPING = {
    "pants-fire": "FAKE",
    "false": "FAKE",
    "barely-true": "FAKE",
    "half-true": "FAKE",
    "mostly-true": "REAL",
    "true": "REAL"
}


LABEL_TO_ID = {
    "FAKE": 0,
    "REAL": 1
}


ID_TO_LABEL = {
    0: "FAKE",
    1: "REAL"
}


ORIGINAL_LABEL_ORDER = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true"
]


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(title):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# LOAD ORIGINAL LIAR DATASET
# ============================================================

def load_original_dataset():

    print_header("LOADING ORIGINAL LIAR TEST DATASET")

    if not ORIGINAL_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Original test dataset not found:\n"
            f"{ORIGINAL_TEST_PATH}"
        )

    df = pd.read_csv(
        ORIGINAL_TEST_PATH,
        sep="\t",
        header=None
    )

    print(f"Original test samples: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if len(df.columns) != 14:
        raise ValueError(
            f"Expected 14 columns, found {len(df.columns)}"
        )

    result = pd.DataFrame({
        "statement": df[STATEMENT_COLUMN],
        "original_label": df[LABEL_COLUMN]
    })

    result["binary_label"] = (
        result["original_label"]
        .map(BINARY_MAPPING)
    )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    if result["original_label"].isna().any():
        raise ValueError(
            "Original labels contain missing values."
        )

    if result["binary_label"].isna().any():

        invalid = (
            result.loc[
                result["binary_label"].isna(),
                "original_label"
            ]
            .unique()
        )

        raise ValueError(
            f"Unknown original labels found: {invalid}"
        )

    # --------------------------------------------------------
    # Clean statements
    # --------------------------------------------------------

    result["statement"] = (
        result["statement"]
        .astype(str)
        .str.strip()
    )

    print()
    print("Original label distribution:")

    print(
        result["original_label"]
        .value_counts()
        .reindex(ORIGINAL_LABEL_ORDER)
    )

    print()
    print("Binary label distribution:")

    print(
        result["binary_label"]
        .value_counts()
        .reindex(["FAKE", "REAL"])
    )

    return result


# ============================================================
# LOAD TOKENIZED BERT DATASET
# ============================================================

def load_bert_dataset():

    print_header("LOADING TOKENIZED BERT TEST DATASET")

    if not BERT_TEST_DIR.exists():
        raise FileNotFoundError(
            f"BERT test dataset not found:\n"
            f"{BERT_TEST_DIR}"
        )

    dataset = load_from_disk(
        str(BERT_TEST_DIR)
    )

    print(f"BERT test samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    required_columns = [
        "labels",
        "input_ids",
        "attention_mask"
    ]

    for column in required_columns:

        if column not in dataset.column_names:

            raise ValueError(
                f"Missing required column: {column}"
            )

    return dataset


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print_header("LOADING TRAINED BERT MODEL")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Trained BERT model not found:\n"
            f"{MODEL_DIR}"
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(DEVICE)
    model.eval()

    print(f"Model: {MODEL_NAME}")
    print(
        f"Number of labels: "
        f"{model.config.num_labels}"
    )

    print(f"Device: {DEVICE}")

    print()
    print("Model label mapping:")

    for key, value in model.config.id2label.items():

        print(f"{key}: {value}")

    return model


# ============================================================
# VALIDATE ALIGNMENT
# ============================================================

def validate_alignment(
    original_df,
    bert_dataset
):

    print_header("VALIDATING DATASET ALIGNMENT")

    if len(original_df) != len(bert_dataset):

        raise ValueError(
            "Dataset size mismatch!\n"
            f"Original: {len(original_df)}\n"
            f"BERT: {len(bert_dataset)}"
        )

    print(
        f"Original dataset samples: {len(original_df)}"
    )

    print(
        f"BERT dataset samples: {len(bert_dataset)}"
    )

    print()
    print(
        "Dataset sizes match."
    )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    original_labels = (
        original_df["binary_label"]
        .map(LABEL_TO_ID)
        .tolist()
    )

    bert_labels = bert_dataset["labels"]

    # Convert to integers
    bert_labels = [
        int(x)
        for x in bert_labels
    ]

    mismatches = []

    for i, (original, bert) in enumerate(
        zip(original_labels, bert_labels)
    ):

        if original != bert:

            mismatches.append(i)

    if mismatches:

        print()
        print(
            f"WARNING: {len(mismatches)} "
            f"label mismatches found."
        )

        print(
            "First mismatches:",
            mismatches[:10]
        )

    else:

        print(
            "Original binary labels and "
            "BERT labels are perfectly aligned."
        )


# ============================================================
# RUN PREDICTIONS
# ============================================================

def run_predictions(
    model,
    dataset
):

    print_header("RUNNING BERT PREDICTIONS")

    all_predictions = []
    all_probabilities = []

    total_samples = len(dataset)

    for start in tqdm(
        range(0, total_samples, BATCH_SIZE),
        desc="Predicting"
    ):

        end = min(
            start + BATCH_SIZE,
            total_samples
        )

        batch = dataset[start:end]

        # ----------------------------------------------------
        # Convert lists to tensors
        # ----------------------------------------------------

        input_ids = torch.tensor(
            batch["input_ids"],
            dtype=torch.long
        ).to(DEVICE)

        attention_mask = torch.tensor(
            batch["attention_mask"],
            dtype=torch.long
        ).to(DEVICE)

        # ----------------------------------------------------
        # token_type_ids
        # ----------------------------------------------------

        token_type_ids = None

        if "token_type_ids" in batch:

            token_type_ids = torch.tensor(
                batch["token_type_ids"],
                dtype=torch.long
            ).to(DEVICE)

        # ----------------------------------------------------
        # Model input
        # ----------------------------------------------------

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        if token_type_ids is not None:

            model_inputs[
                "token_type_ids"
            ] = token_type_ids

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                **model_inputs
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().tolist()
        )

    print("Predictions completed.")

    return (
        all_predictions,
        all_probabilities
    )


# ============================================================
# BUILD RESULTS DATAFRAME
# ============================================================

def build_results(
    original_df,
    predictions,
    probabilities
):

    print_header("BUILDING ERROR ANALYSIS DATAFRAME")

    results = original_df.copy()

    results["true_id"] = (
        results["binary_label"]
        .map(LABEL_TO_ID)
    )

    results["predicted_id"] = predictions

    results["predicted_label"] = [
        ID_TO_LABEL[int(x)]
        for x in predictions
    ]

    results["prob_fake"] = [
        float(prob[0])
        for prob in probabilities
    ]

    results["prob_real"] = [
        float(prob[1])
        for prob in probabilities
    ]

    results["confidence"] = [
        max(prob)
        for prob in probabilities
    ]

    results["correct"] = (
        results["true_id"]
        == results["predicted_id"]
    )

    # --------------------------------------------------------
    # Error type
    # --------------------------------------------------------

    def get_error_type(row):

        if row["correct"]:
            return "CORRECT"

        if (
            row["binary_label"] == "FAKE"
            and row["predicted_label"] == "REAL"
        ):
            return "FAKE_TO_REAL"

        if (
            row["binary_label"] == "REAL"
            and row["predicted_label"] == "FAKE"
        ):
            return "REAL_TO_FAKE"

        return "UNKNOWN"

    results["error_type"] = results.apply(
        get_error_type,
        axis=1
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Token length is added separately by add_token_lengths()
    # --------------------------------------------------------

    return results


# ============================================================
# ADD TOKEN LENGTHS
# ============================================================

def add_token_lengths(
    results,
    bert_dataset
):

    token_lengths = []

    for mask in bert_dataset["attention_mask"]:

        token_lengths.append(
            int(sum(mask))
        )

    results["token_length"] = token_lengths

    return results


# ============================================================
# ORIGINAL LABEL PERFORMANCE
# ============================================================

def analyze_original_labels(results):

    print_header("PERFORMANCE BY ORIGINAL LIAR LABEL")

    rows = []

    for label in ORIGINAL_LABEL_ORDER:

        subset = results[
            results["original_label"] == label
        ]

        if len(subset) == 0:
            continue

        accuracy = (
            subset["correct"].mean()
        )

        rows.append({
            "original_label": label,
            "samples": len(subset),
            "correct": int(
                subset["correct"].sum()
            ),
            "incorrect": int(
                (~subset["correct"]).sum()
            ),
            "accuracy": accuracy
        })

    df = pd.DataFrame(rows)

    print(
        df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format
            }
        )
    )

    return df


# ============================================================
# ORIGINAL LABEL × PREDICTION
# ============================================================

def analyze_label_predictions(results):

    print_header(
        "ORIGINAL LABEL VS BERT PREDICTION"
    )

    table = pd.crosstab(
        results["original_label"],
        results["predicted_label"]
    )

    table = table.reindex(
        ORIGINAL_LABEL_ORDER
    )

    print(table)

    return table


# ============================================================
# ERROR RATES BY ORIGINAL LABEL
# ============================================================

def analyze_error_rates(results):

    print_header(
        "ERROR RATES BY ORIGINAL LIAR LABEL"
    )

    rows = []

    for label in ORIGINAL_LABEL_ORDER:

        subset = results[
            results["original_label"] == label
        ]

        total = len(subset)

        incorrect = (
            ~subset["correct"]
        ).sum()

        error_rate = (
            incorrect / total
            if total > 0
            else 0
        )

        rows.append({
            "original_label": label,
            "samples": total,
            "errors": int(incorrect),
            "error_rate": error_rate
        })

    df = pd.DataFrame(rows)

    print(
        df.to_string(
            index=False,
            formatters={
                "error_rate": "{:.4f}".format
            }
        )
    )

    return df


# ============================================================
# CONFIDENCE BY ORIGINAL LABEL
# ============================================================

def analyze_confidence(results):

    print_header(
        "CONFIDENCE BY ORIGINAL LIAR LABEL"
    )

    summary = (
        results
        .groupby("original_label")
        .agg(
            samples=("original_label", "size"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median")
        )
        .reindex(ORIGINAL_LABEL_ORDER)
    )

    print(
        summary.to_string(
            formatters={
                "mean_confidence": "{:.4f}".format,
                "median_confidence": "{:.4f}".format
            }
        )
    )

    return summary


# ============================================================
# HIGH CONFIDENCE ERRORS
# ============================================================

def analyze_high_confidence_errors(results):

    print_header(
        "HIGH-CONFIDENCE ERRORS BY ORIGINAL LABEL"
    )

    errors = results[
        ~results["correct"]
    ].copy()

    high_confidence = errors[
        errors["confidence"] >= 0.90
    ]

    print(
        f"Total errors: {len(errors)}"
    )

    print(
        f"High-confidence errors >= 0.90: "
        f"{len(high_confidence)}"
    )

    print()
    print("High-confidence errors by original label:")

    counts = (
        high_confidence[
            "original_label"
        ]
        .value_counts()
        .reindex(
            ORIGINAL_LABEL_ORDER,
            fill_value=0
        )
    )

    print(counts)

    return high_confidence


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    original_performance,
    label_predictions,
    error_rates,
    confidence_summary,
    high_confidence_errors
):

    print_header("SAVING ANALYSIS FILES")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # All predictions
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_DIR / "all_predictions_original_labels.csv",
        index=False
    )

    # --------------------------------------------------------
    # Misclassified
    # --------------------------------------------------------

    results[
        ~results["correct"]
    ].to_csv(
        OUTPUT_DIR / "misclassified_original_labels.csv",
        index=False
    )

    # --------------------------------------------------------
    # High confidence errors
    # --------------------------------------------------------

    high_confidence_errors.to_csv(
        OUTPUT_DIR / "high_confidence_errors.csv",
        index=False
    )

    # --------------------------------------------------------
    # Original label performance
    # --------------------------------------------------------

    original_performance.to_csv(
        OUTPUT_DIR / "performance_by_original_label.csv",
        index=False
    )

    # --------------------------------------------------------
    # Label prediction table
    # --------------------------------------------------------

    label_predictions.to_csv(
        OUTPUT_DIR / "original_label_prediction_table.csv"
    )

    # --------------------------------------------------------
    # Error rates
    # --------------------------------------------------------

    error_rates.to_csv(
        OUTPUT_DIR / "error_rates_by_original_label.csv",
        index=False
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_summary.to_csv(
        OUTPUT_DIR / "confidence_by_original_label.csv"
    )

    print(
        f"Saved results to:\n{OUTPUT_DIR}"
    )


# ============================================================
# PLOT ACCURACY BY ORIGINAL LABEL
# ============================================================

def plot_accuracy_by_label(
    performance
):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        performance["original_label"],
        performance["accuracy"]
    )

    plt.ylim(
        0,
        1
    )

    plt.xlabel(
        "Original LIAR label"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "BERT Accuracy by Original LIAR Label"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "accuracy_by_original_label.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved plot: {path}"
    )


# ============================================================
# PLOT ERROR RATE
# ============================================================

def plot_error_rate(
    error_rates
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        error_rates["original_label"],
        error_rates["error_rate"]
    )

    plt.ylim(
        0,
        1
    )

    plt.xlabel(
        "Original LIAR label"
    )

    plt.ylabel(
        "Error rate"
    )

    plt.title(
        "BERT Error Rate by Original LIAR Label"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "error_rate_by_original_label.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved plot: {path}"
    )


# ============================================================
# PLOT CONFIDENCE
# ============================================================

def plot_confidence(
    results
):

    plt.figure(
        figsize=(10, 6)
    )

    data = [
        results[
            results["original_label"] == label
        ]["confidence"].values
        for label in ORIGINAL_LABEL_ORDER
    ]

    plt.boxplot(
        data,
        tick_labels=ORIGINAL_LABEL_ORDER
    )

    plt.xlabel(
        "Original LIAR label"
    )

    plt.ylabel(
        "BERT confidence"
    )

    plt.title(
        "BERT Confidence by Original LIAR Label"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "confidence_by_original_label.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved plot: {path}"
    )


# ============================================================
# PRINT IMPORTANT ERRORS
# ============================================================

def print_examples(
    results,
    n=20
):

    print_header(
        "HIGH-CONFIDENCE ERROR EXAMPLES"
    )

    errors = (
        results[
            ~results["correct"]
        ]
        .sort_values(
            "confidence",
            ascending=False
        )
        .head(n)
    )

    for _, row in errors.iterrows():

        print("-" * 60)

        print(
            f"Original LIAR label: "
            f"{row['original_label']}"
        )

        print(
            f"Binary true label: "
            f"{row['binary_label']}"
        )

        print(
            f"BERT prediction: "
            f"{row['predicted_label']}"
        )

        print(
            f"Confidence: "
            f"{row['confidence']:.4f}"
        )

        print(
            f"Probability FAKE: "
            f"{row['prob_fake']:.4f}"
        )

        print(
            f"Probability REAL: "
            f"{row['prob_real']:.4f}"
        )

        print(
            f"Token length: "
            f"{row['token_length']}"
        )

        print(
            f"Statement:\n"
            f"{row['statement']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "BERT ORIGINAL LIAR LABEL ERROR ANALYSIS"
    )

    print()
    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    original_df = load_original_dataset()

    # --------------------------------------------------------
    # Load BERT dataset
    # --------------------------------------------------------

    bert_dataset = load_bert_dataset()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Validate alignment
    # --------------------------------------------------------

    validate_alignment(
        original_df,
        bert_dataset
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions, probabilities = run_predictions(
        model,
        bert_dataset
    )

    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    results = build_results(
        original_df,
        predictions,
        probabilities
    )

    # --------------------------------------------------------
    # Add actual token lengths
    # --------------------------------------------------------

    results = add_token_lengths(
        results,
        bert_dataset
    )

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    original_performance = (
        analyze_original_labels(
            results
        )
    )

    label_predictions = (
        analyze_label_predictions(
            results
        )
    )

    error_rates = (
        analyze_error_rates(
            results
        )
    )

    confidence_summary = (
        analyze_confidence(
            results
        )
    )

    high_confidence_errors = (
        analyze_high_confidence_errors(
            results
        )
    )

    # --------------------------------------------------------
    # Print examples
    # --------------------------------------------------------

    print_examples(
        results,
        n=20
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        original_performance,
        label_predictions,
        error_rates,
        confidence_summary,
        high_confidence_errors
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print_header(
        "GENERATING PLOTS"
    )

    plot_accuracy_by_label(
        original_performance
    )

    plot_error_rate(
        error_rates
    )

    plot_confidence(
        results
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_header(
        "ORIGINAL LABEL ERROR ANALYSIS COMPLETED"
    )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print("Generated files:")

    print(
        "  - all_predictions_original_labels.csv"
    )

    print(
        "  - misclassified_original_labels.csv"
    )

    print(
        "  - high_confidence_errors.csv"
    )

    print(
        "  - performance_by_original_label.csv"
    )

    print(
        "  - original_label_prediction_table.csv"
    )

    print(
        "  - error_rates_by_original_label.csv"
    )

    print(
        "  - confidence_by_original_label.csv"
    )

    print(
        "  - plots/accuracy_by_original_label.png"
    )

    print(
        "  - plots/error_rate_by_original_label.png"
    )

    print(
        "  - plots/confidence_by_original_label.png"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()