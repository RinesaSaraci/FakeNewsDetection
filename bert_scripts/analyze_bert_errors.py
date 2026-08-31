import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path("models/bert/binary/final")
TEST_DATASET_DIR = Path("data/processed/bert/binary/test")

OUTPUT_DIR = Path("results/bert/error_analysis")
PLOTS_DIR = OUTPUT_DIR / "plots"

MODEL_NAME = "bert-base-uncased"

LABEL_NAMES = {
    0: "FAKE",
    1: "REAL"
}

MAX_LENGTH = 128

TOP_N = 20


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """Select CUDA if available, otherwise CPU."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available: YES")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA available: NO")
        print("Using CPU.")

    return device


# ============================================================
# LOAD DATASET
# ============================================================

def load_test_dataset():
    """Load the tokenized BERT test dataset."""

    print("=" * 60)
    print("LOADING TEST DATASET")
    print("=" * 60)

    if not TEST_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_DATASET_DIR}"
        )

    dataset = load_from_disk(str(TEST_DATASET_DIR))

    print(f"Test samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    required_columns = [
        "labels",
        "input_ids",
        "attention_mask"
    ]

    for column in required_columns:
        if column not in dataset.column_names:
            raise ValueError(
                f"Required column '{column}' is missing."
            )

    print("Test dataset loaded successfully.")

    return dataset


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):
    """Load the trained BERT sequence classification model."""

    print("\n" + "=" * 60)
    print("LOADING TRAINED BERT MODEL")
    print("=" * 60)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_DIR}"
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    model.to(device)

    model.eval()

    print("Model loaded successfully.")
    print(f"Number of labels: {model.config.num_labels}")

    print("\nLabel mapping:")

    for label_id, label_name in LABEL_NAMES.items():
        print(f"{label_id}: {label_name}")

    return model, tokenizer


# ============================================================
# RUN PREDICTIONS
# ============================================================

def run_predictions(dataset, model, device):
    """Generate predictions and probabilities."""

    print("\n" + "=" * 60)
    print("RUNNING PREDICTIONS")
    print("=" * 60)

    trainer = Trainer(
        model=model
    )

    predictions = trainer.predict(dataset)

    logits = predictions.predictions
    true_labels = predictions.label_ids

    # Convert logits to probabilities
    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1
    ).numpy()

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    print("Predictions completed.")

    return (
        true_labels,
        predicted_labels,
        probabilities,
        confidence
    )


# ============================================================
# EXTRACT STATEMENTS
# ============================================================

def get_original_statements(dataset, tokenizer):
    """
    Decode tokenized inputs back into readable statements.
    """

    statements = []

    for input_ids in dataset["input_ids"]:

        text = tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        statements.append(text)

    return statements


# ============================================================
# TOKEN LENGTH
# ============================================================

def calculate_token_lengths(dataset):
    """Calculate the number of real tokens for every sample."""

    lengths = []

    for attention_mask in dataset["attention_mask"]:

        length = int(
            sum(attention_mask)
        )

        lengths.append(length)

    return lengths


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

def create_results_dataframe(
    dataset,
    tokenizer,
    true_labels,
    predicted_labels,
    probabilities,
    confidence
):
    """Create a detailed prediction DataFrame."""

    statements = get_original_statements(
        dataset,
        tokenizer
    )

    token_lengths = calculate_token_lengths(
        dataset
    )

    df = pd.DataFrame({
        "statement": statements,

        "true_label_id": true_labels,

        "true_label": [
            LABEL_NAMES[int(x)]
            for x in true_labels
        ],

        "predicted_label_id": predicted_labels,

        "predicted_label": [
            LABEL_NAMES[int(x)]
            for x in predicted_labels
        ],

        "prob_fake": probabilities[:, 0],

        "prob_real": probabilities[:, 1],

        "confidence": confidence,

        "correct": (
            true_labels == predicted_labels
        ),

        "token_length": token_lengths
    })

    # Prediction direction
    def get_error_type(row):

        if row["correct"]:
            return "CORRECT"

        if (
            row["true_label"] == "FAKE"
            and row["predicted_label"] == "REAL"
        ):
            return "FAKE_TO_REAL"

        if (
            row["true_label"] == "REAL"
            and row["predicted_label"] == "FAKE"
        ):
            return "REAL_TO_FAKE"

        return "OTHER"

    df["error_type"] = df.apply(
        get_error_type,
        axis=1
    )

    return df


# ============================================================
# OVERALL ERROR ANALYSIS
# ============================================================

def print_overall_analysis(df):

    print("\n" + "=" * 60)
    print("OVERALL ERROR ANALYSIS")
    print("=" * 60)

    total = len(df)

    correct = int(
        df["correct"].sum()
    )

    incorrect = total - correct

    accuracy = correct / total

    print(f"Total samples:       {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Accuracy:             {accuracy:.4f}")

    print("\nError distribution:")

    print(
        df["error_type"]
        .value_counts()
        .to_string()
    )

    print("\nPrediction distribution:")

    print(
        df["predicted_label"]
        .value_counts()
        .to_string()
    )


# ============================================================
# ERROR DIRECTION ANALYSIS
# ============================================================

def analyze_error_directions(df):

    print("\n" + "=" * 60)
    print("ERROR DIRECTION ANALYSIS")
    print("=" * 60)

    fake_to_real = df[
        df["error_type"] == "FAKE_TO_REAL"
    ]

    real_to_fake = df[
        df["error_type"] == "REAL_TO_FAKE"
    ]

    print(
        f"FAKE -> REAL errors: "
        f"{len(fake_to_real)}"
    )

    print(
        f"REAL -> FAKE errors: "
        f"{len(real_to_fake)}"
    )

    print("\nFAKE -> REAL confidence:")
    if len(fake_to_real) > 0:
        print(
            f"Mean:   {fake_to_real['confidence'].mean():.4f}"
        )
        print(
            f"Median: {fake_to_real['confidence'].median():.4f}"
        )

    print("\nREAL -> FAKE confidence:")
    if len(real_to_fake) > 0:
        print(
            f"Mean:   {real_to_fake['confidence'].mean():.4f}"
        )
        print(
            f"Median: {real_to_fake['confidence'].median():.4f}"
        )


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

def analyze_confidence(df):

    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)

    correct = df[
        df["correct"] == True
    ]

    incorrect = df[
        df["correct"] == False
    ]

    print("\nCorrect predictions:")
    print(
        f"Count:  {len(correct)}"
    )
    print(
        f"Mean:   {correct['confidence'].mean():.4f}"
    )
    print(
        f"Median: {correct['confidence'].median():.4f}"
    )

    print("\nIncorrect predictions:")
    print(
        f"Count:  {len(incorrect)}"
    )
    print(
        f"Mean:   {incorrect['confidence'].mean():.4f}"
    )
    print(
        f"Median: {incorrect['confidence'].median():.4f}"
    )

    print("\nHigh-confidence errors:")

    high_confidence = incorrect[
        incorrect["confidence"] >= 0.90
    ]

    print(
        f"Errors with confidence >= 0.90: "
        f"{len(high_confidence)}"
    )

    high_confidence_95 = incorrect[
        incorrect["confidence"] >= 0.95
    ]

    print(
        f"Errors with confidence >= 0.95: "
        f"{len(high_confidence_95)}"
    )


# ============================================================
# LENGTH ANALYSIS
# ============================================================

def analyze_by_length(df):

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS BY TOKEN LENGTH")
    print("=" * 60)

    bins = [
        0,
        16,
        32,
        64,
        128
    ]

    labels = [
        "1-16",
        "17-32",
        "33-64",
        "65-128"
    ]

    df["length_group"] = pd.cut(
        df["token_length"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    grouped = (
        df
        .groupby(
            "length_group",
            observed=False
        )
        .agg(
            samples=("correct", "size"),
            correct=("correct", "sum")
        )
    )

    grouped["incorrect"] = (
        grouped["samples"]
        - grouped["correct"]
    )

    grouped["accuracy"] = (
        grouped["correct"]
        / grouped["samples"]
    )

    grouped["error_rate"] = (
        grouped["incorrect"]
        / grouped["samples"]
    )

    print(
        grouped.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return grouped


# ============================================================
# HIGH-CONFIDENCE ERRORS
# ============================================================

def get_high_confidence_errors(df):

    errors = df[
        df["correct"] == False
    ].copy()

    errors = errors.sort_values(
        "confidence",
        ascending=False
    )

    return errors


# ============================================================
# LOW-CONFIDENCE ERRORS
# ============================================================

def get_low_confidence_errors(df):

    errors = df[
        df["correct"] == False
    ].copy()

    errors = errors.sort_values(
        "confidence",
        ascending=True
    )

    return errors


# ============================================================
# PRINT EXAMPLES
# ============================================================

def print_error_examples(df):

    print("\n" + "=" * 60)
    print("HIGH-CONFIDENCE ERROR EXAMPLES")
    print("=" * 60)

    errors = get_high_confidence_errors(
        df
    ).head(TOP_N)

    for index, row in errors.iterrows():

        print("\n" + "-" * 60)

        print(
            f"True label:       {row['true_label']}"
        )

        print(
            f"Predicted label:  {row['predicted_label']}"
        )

        print(
            f"Confidence:       {row['confidence']:.4f}"
        )

        print(
            f"Probability FAKE: {row['prob_fake']:.4f}"
        )

        print(
            f"Probability REAL: {row['prob_real']:.4f}"
        )

        print(
            f"Token length:     {row['token_length']}"
        )

        print(
            f"Error type:       {row['error_type']}"
        )

        print(
            f"Statement:\n{row['statement']}"
        )


# ============================================================
# SAVE CSV FILES
# ============================================================

def save_csv_results(df):

    print("\n" + "=" * 60)
    print("SAVING ERROR ANALYSIS FILES")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # All predictions
    df.to_csv(
        OUTPUT_DIR / "all_predictions.csv",
        index=False
    )

    print("Saved: all_predictions.csv")

    # All errors
    errors = df[
        df["correct"] == False
    ]

    errors.to_csv(
        OUTPUT_DIR / "misclassified.csv",
        index=False
    )

    print("Saved: misclassified.csv")

    # False negatives
    false_negatives = df[
        df["error_type"] == "FAKE_TO_REAL"
    ]

    false_negatives.to_csv(
        OUTPUT_DIR / "false_negatives.csv",
        index=False
    )

    print("Saved: false_negatives.csv")

    # False positives
    false_positives = df[
        df["error_type"] == "REAL_TO_FAKE"
    ]

    false_positives.to_csv(
        OUTPUT_DIR / "false_positives.csv",
        index=False
    )

    print("Saved: false_positives.csv")

    # High confidence errors
    high_confidence = (
        get_high_confidence_errors(df)
    )

    high_confidence.to_csv(
        OUTPUT_DIR / "high_confidence_errors.csv",
        index=False
    )

    print(
        "Saved: high_confidence_errors.csv"
    )

    # Low confidence errors
    low_confidence = (
        get_low_confidence_errors(df)
    )

    low_confidence.to_csv(
        OUTPUT_DIR / "low_confidence_errors.csv",
        index=False
    )

    print(
        "Saved: low_confidence_errors.csv"
    )


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(df):

    from sklearn.metrics import confusion_matrix

    true_labels = df["true_label_id"]
    predicted_labels = df["predicted_label_id"]

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_title(
        "BERT Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["FAKE", "REAL"]
    )

    ax.set_yticklabels(
        ["FAKE", "REAL"]
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved plot: {path}")


# ============================================================
# PLOT CONFIDENCE DISTRIBUTION
# ============================================================

def plot_confidence_distribution(df):

    correct = df[
        df["correct"] == True
    ]["confidence"]

    incorrect = df[
        df["correct"] == False
    ]["confidence"]

    plt.figure()

    plt.hist(
        correct,
        bins=20,
        alpha=0.6,
        label="Correct"
    )

    plt.hist(
        incorrect,
        bins=20,
        alpha=0.6,
        label="Incorrect"
    )

    plt.xlabel(
        "Prediction Confidence"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "BERT Prediction Confidence"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "confidence_distribution.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved plot: {path}")


# ============================================================
# PLOT ACCURACY BY LENGTH
# ============================================================

def plot_accuracy_by_length(grouped):

    plt.figure()

    x = np.arange(
        len(grouped.index)
    )

    accuracy = (
        grouped["accuracy"]
        .values
    )

    plt.bar(
        x,
        accuracy
    )

    plt.xticks(
        x,
        grouped.index.astype(str)
    )

    plt.xlabel(
        "Token Length"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "BERT Accuracy by Token Length"
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    path = (
        PLOTS_DIR
        / "accuracy_by_length.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved plot: {path}")


# ============================================================
# CREATE TEXT REPORT
# ============================================================

def create_text_report(
    df,
    length_analysis
):

    total = len(df)

    correct = int(
        df["correct"].sum()
    )

    incorrect = total - correct

    accuracy = (
        correct / total
    )

    fake_to_real = len(
        df[
            df["error_type"]
            == "FAKE_TO_REAL"
        ]
    )

    real_to_fake = len(
        df[
            df["error_type"]
            == "REAL_TO_FAKE"
        ]
    )

    incorrect_df = df[
        df["correct"] == False
    ]

    report_lines = []

    report_lines.append(
        "BERT ERROR ANALYSIS REPORT"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append(
        f"Model: {MODEL_NAME}"
    )

    report_lines.append(
        f"Test samples: {total}"
    )

    report_lines.append(
        f"Correct predictions: {correct}"
    )

    report_lines.append(
        f"Incorrect predictions: {incorrect}"
    )

    report_lines.append(
        f"Accuracy: {accuracy:.4f}"
    )

    report_lines.append("")
    report_lines.append(
        "ERROR DIRECTIONS"
    )

    report_lines.append(
        f"FAKE -> REAL: {fake_to_real}"
    )

    report_lines.append(
        f"REAL -> FAKE: {real_to_fake}"
    )

    if len(incorrect_df) > 0:

        report_lines.append("")
        report_lines.append(
            "CONFIDENCE"
        )

        report_lines.append(
            f"Mean confidence - correct: "
            f"{df[df['correct']]['confidence'].mean():.4f}"
        )

        report_lines.append(
            f"Mean confidence - incorrect: "
            f"{incorrect_df['confidence'].mean():.4f}"
        )

        report_lines.append(
            f"High-confidence errors >= 0.90: "
            f"{len(incorrect_df[incorrect_df['confidence'] >= 0.90])}"
        )

        report_lines.append(
            f"High-confidence errors >= 0.95: "
            f"{len(incorrect_df[incorrect_df['confidence'] >= 0.95])}"
        )

    report_lines.append("")
    report_lines.append(
        "ACCURACY BY TOKEN LENGTH"
    )

    report_lines.append(
        length_analysis.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    report_path = (
        OUTPUT_DIR
        / "error_analysis.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report_lines)
        )

    print(
        f"Saved report: {report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BERT ERROR ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_test_dataset()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, tokenizer = load_model(
        device
    )

    # --------------------------------------------------------
    # Run predictions
    # --------------------------------------------------------

    (
        true_labels,
        predicted_labels,
        probabilities,
        confidence
    ) = run_predictions(
        dataset,
        model,
        device
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = create_results_dataframe(
        dataset,
        tokenizer,
        true_labels,
        predicted_labels,
        probabilities,
        confidence
    )

    # --------------------------------------------------------
    # Overall analysis
    # --------------------------------------------------------

    print_overall_analysis(
        df
    )

    # --------------------------------------------------------
    # Error directions
    # --------------------------------------------------------

    analyze_error_directions(
        df
    )

    # --------------------------------------------------------
    # Confidence analysis
    # --------------------------------------------------------

    analyze_confidence(
        df
    )

    # --------------------------------------------------------
    # Token length analysis
    # --------------------------------------------------------

    length_analysis = analyze_by_length(
        df
    )

    # --------------------------------------------------------
    # Error examples
    # --------------------------------------------------------

    print_error_examples(
        df
    )

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    save_csv_results(
        df
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_confusion_matrix(
        df
    )

    plot_confidence_distribution(
        df
    )

    plot_accuracy_by_length(
        length_analysis
    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    create_text_report(
        df,
        length_analysis
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETED")
    print("=" * 60)

    print("\nResults saved to:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    print(
        "  - all_predictions.csv"
    )

    print(
        "  - misclassified.csv"
    )

    print(
        "  - false_positives.csv"
    )

    print(
        "  - false_negatives.csv"
    )

    print(
        "  - high_confidence_errors.csv"
    )

    print(
        "  - low_confidence_errors.csv"
    )

    print(
        "  - error_analysis.txt"
    )

    print(
        "  - plots/confusion_matrix.png"
    )

    print(
        "  - plots/confidence_distribution.png"
    )

    print(
        "  - plots/accuracy_by_length.png"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()