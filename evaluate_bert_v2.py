import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"models\bert\binary_v2"

TEST_DATA_PATH = r"data\processed\bert\binary\test"

OUTPUT_DIR = r"results\bert\binary_v2"

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

MODEL_NAME = "bert-base-uncased"

NUM_LABELS = 2

LABEL_NAMES = {
    0: "FAKE",
    1: "REAL"
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(title):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    print_header("LOADING TEST DATASET")

    print(f"Path: {TEST_DATA_PATH}")

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_DATA_PATH}"
        )

    test_dataset = load_from_disk(TEST_DATA_PATH)

    print(f"Test samples: {len(test_dataset)}")

    print("\nDataset columns:")
    print(test_dataset.column_names)

    required_columns = [
        "labels",
        "input_ids",
        "attention_mask"
    ]

    for column in required_columns:

        if column not in test_dataset.column_names:

            raise ValueError(
                f"Required column '{column}' "
                f"is missing from test dataset."
            )

    print("\nTest dataset loaded successfully.")

    return test_dataset


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(test_dataset):

    print_header("VALIDATING TEST DATASET")

    labels = np.array(test_dataset["labels"])

    print(f"Number of samples: {len(labels)}")

    unique_labels = np.unique(labels)

    print(f"Unique labels: {unique_labels}")

    if not set(unique_labels).issubset({0, 1}):

        raise ValueError(
            f"Unexpected labels found: {unique_labels}"
        )

    fake_count = int(np.sum(labels == 0))
    real_count = int(np.sum(labels == 1))

    print()
    print(f"FAKE samples: {fake_count}")
    print(f"REAL samples: {real_count}")

    print("\nDataset validation passed.")


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    print_header("LOADING BERT TOKENIZER")

    print(f"Tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"Maximum sequence length: "
        f"{tokenizer.model_max_length}"
    )

    print("Tokenizer loaded successfully.")

    return tokenizer


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print_header("LOADING TRAINED BERT V2 MODEL")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    print(f"Model path: {MODEL_PATH}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_LABELS
    )

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")

    print(
        f"Number of labels: "
        f"{model.config.num_labels}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print("\nModel label mapping:")

    for label_id, label_name in LABEL_NAMES.items():

        print(f"{label_id}: {label_name}")

    return model


# ============================================================
# CREATE TRAINER
# ============================================================

def create_trainer(model):

    print_header("CREATING EVALUATION TRAINER")

    trainer = Trainer(
        model=model
    )

    print("Trainer created successfully.")

    return trainer


# ============================================================
# RUN PREDICTIONS
# ============================================================

def run_predictions(trainer, test_dataset):

    print_header("RUNNING BERT V2 PREDICTIONS")

    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits)

    # Softmax
    exp_logits = np.exp(
        logits - np.max(logits, axis=1, keepdims=True)
    )

    probabilities = (
        exp_logits /
        np.sum(exp_logits, axis=1, keepdims=True)
    )

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
        predicted_labels,
        probabilities,
        confidence
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels
):

    print_header("OVERALL PERFORMANCE")

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

    return metrics


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def create_classification_report(
    true_labels,
    predicted_labels
):

    print_header("CLASSIFICATION REPORT")

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["FAKE", "REAL"],
        digits=4,
        zero_division=0
    )

    print(report)

    return report


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    true_labels,
    predicted_labels
):

    print_header("CONFUSION MATRIX")

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )

    print("Rows = True labels")
    print("Columns = Predicted labels")

    print()
    print("Labels:")
    print("[FAKE, REAL]")

    print()
    print(cm)

    # Save matrix as CSV

    cm_df = pd.DataFrame(
        cm,
        index=["FAKE", "REAL"],
        columns=["FAKE", "REAL"]
    )

    cm_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    # Plot

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["FAKE", "REAL"]
    )

    disp.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        "BERT V2 Confusion Matrix"
    )

    plt.tight_layout()

    plot_path = os.path.join(
        PLOTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(f"Saved matrix: {cm_path}")
    print(f"Saved plot:   {plot_path}")

    return cm


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

def analyze_prediction_distribution(
    true_labels,
    predicted_labels
):

    print_header("PREDICTION DISTRIBUTION")

    true_fake = int(
        np.sum(true_labels == 0)
    )

    true_real = int(
        np.sum(true_labels == 1)
    )

    pred_fake = int(
        np.sum(predicted_labels == 0)
    )

    pred_real = int(
        np.sum(predicted_labels == 1)
    )

    print("True labels:")
    print(f"FAKE: {true_fake}")
    print(f"REAL: {true_real}")

    print()

    print("Predicted labels:")
    print(f"FAKE: {pred_fake}")
    print(f"REAL: {pred_real}")

    distribution = {
        "true_fake": true_fake,
        "true_real": true_real,
        "predicted_fake": pred_fake,
        "predicted_real": pred_real
    }

    return distribution


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

def analyze_confidence(
    true_labels,
    predicted_labels,
    confidence
):

    print_header("CONFIDENCE ANALYSIS")

    correct_mask = (
        true_labels == predicted_labels
    )

    incorrect_mask = (
        true_labels != predicted_labels
    )

    correct_confidence = confidence[
        correct_mask
    ]

    incorrect_confidence = confidence[
        incorrect_mask
    ]

    print("Correct predictions:")
    print(
        f"Count:  {len(correct_confidence)}"
    )

    if len(correct_confidence) > 0:

        print(
            f"Mean:   "
            f"{np.mean(correct_confidence):.4f}"
        )

        print(
            f"Median: "
            f"{np.median(correct_confidence):.4f}"
        )

    print()

    print("Incorrect predictions:")
    print(
        f"Count:  {len(incorrect_confidence)}"
    )

    if len(incorrect_confidence) > 0:

        print(
            f"Mean:   "
            f"{np.mean(incorrect_confidence):.4f}"
        )

        print(
            f"Median: "
            f"{np.median(incorrect_confidence):.4f}"
        )

    high_90 = int(
        np.sum(
            incorrect_confidence >= 0.90
        )
    )

    high_95 = int(
        np.sum(
            incorrect_confidence >= 0.95
        )
    )

    print()

    print(
        f"High-confidence errors >= 0.90: "
        f"{high_90}"
    )

    print(
        f"High-confidence errors >= 0.95: "
        f"{high_95}"
    )

    # Confidence plot

    plt.figure(figsize=(8, 6))

    plt.hist(
        correct_confidence,
        bins=20,
        alpha=0.6,
        label="Correct"
    )

    plt.hist(
        incorrect_confidence,
        bins=20,
        alpha=0.6,
        label="Incorrect"
    )

    plt.xlabel("Confidence")
    plt.ylabel("Number of Samples")
    plt.title(
        "BERT V2 Prediction Confidence"
    )

    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join(
        PLOTS_DIR,
        "confidence_distribution.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved plot: {plot_path}"
    )

    return {
        "correct_count": int(
            len(correct_confidence)
        ),
        "incorrect_count": int(
            len(incorrect_confidence)
        ),
        "correct_mean": float(
            np.mean(correct_confidence)
        ) if len(correct_confidence) else 0.0,
        "incorrect_mean": float(
            np.mean(incorrect_confidence)
        ) if len(incorrect_confidence) else 0.0,
        "high_confidence_90": high_90,
        "high_confidence_95": high_95
    }


# ============================================================
# BUILD PREDICTION DATAFRAME
# ============================================================

def build_predictions_dataframe(
    test_dataset,
    true_labels,
    predicted_labels,
    probabilities,
    confidence
):

    print_header("BUILDING PREDICTION DATAFRAME")

    df = pd.DataFrame({
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
        "probability_fake": probabilities[:, 0],
        "probability_real": probabilities[:, 1],
        "confidence": confidence,
        "correct": (
            true_labels == predicted_labels
        )
    })

    # Add token length

    token_lengths = []

    for ids, mask in zip(
        test_dataset["input_ids"],
        test_dataset["attention_mask"]
    ):

        ids = np.asarray(ids)
        mask = np.asarray(mask)

        token_length = int(
            np.sum(mask)
        )

        token_lengths.append(
            token_length
        )

    df["token_length"] = token_lengths

    # Error type

    df["error_type"] = "CORRECT"

    fake_to_real = (
        (true_labels == 0) &
        (predicted_labels == 1)
    )

    real_to_fake = (
        (true_labels == 1) &
        (predicted_labels == 0)
    )

    df.loc[
        fake_to_real,
        "error_type"
    ] = "FAKE_TO_REAL"

    df.loc[
        real_to_fake,
        "error_type"
    ] = "REAL_TO_FAKE"

    print(
        f"Dataframe created: "
        f"{len(df)} rows"
    )

    return df


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_prediction_files(df):

    print_header("SAVING PREDICTION FILES")

    all_path = os.path.join(
        OUTPUT_DIR,
        "all_predictions.csv"
    )

    misclassified_path = os.path.join(
        OUTPUT_DIR,
        "misclassified.csv"
    )

    high_confidence_path = os.path.join(
        OUTPUT_DIR,
        "high_confidence_errors.csv"
    )

    df.to_csv(
        all_path,
        index=False
    )

    df[
        df["correct"] == False
    ].to_csv(
        misclassified_path,
        index=False
    )

    df[
        (df["correct"] == False) &
        (df["confidence"] >= 0.90)
    ].to_csv(
        high_confidence_path,
        index=False
    )

    print(f"Saved: {all_path}")
    print(f"Saved: {misclassified_path}")
    print(
        f"Saved: {high_confidence_path}"
    )


# ============================================================
# ERROR DIRECTION ANALYSIS
# ============================================================

def analyze_error_direction(df):

    print_header("ERROR DIRECTION ANALYSIS")

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

    if len(fake_to_real) > 0:

        print("\nFAKE -> REAL confidence:")

        print(
            f"Mean:   "
            f"{fake_to_real['confidence'].mean():.4f}"
        )

        print(
            f"Median: "
            f"{fake_to_real['confidence'].median():.4f}"
        )

    if len(real_to_fake) > 0:

        print("\nREAL -> FAKE confidence:")

        print(
            f"Mean:   "
            f"{real_to_fake['confidence'].mean():.4f}"
        )

        print(
            f"Median: "
            f"{real_to_fake['confidence'].median():.4f}"
        )


# ============================================================
# SAVE METRICS JSON
# ============================================================

def save_metrics(
    metrics,
    confidence_stats,
    distribution
):

    metrics_data = {
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "test_samples": 1267,
        "device": str(DEVICE),
        "metrics": metrics,
        "confidence": confidence_stats,
        "distribution": distribution
    }

    path = os.path.join(
        OUTPUT_DIR,
        "metrics.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics_data,
            f,
            indent=4
        )

    print(
        f"Saved metrics: {path}"
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

def save_text_report(
    metrics,
    classification_report_text,
    cm,
    distribution,
    confidence_stats
):

    path = os.path.join(
        OUTPUT_DIR,
        "evaluation_report.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BERT V2 EVALUATION REPORT\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Model: {MODEL_NAME}\n"
        )

        f.write(
            f"Model path: {MODEL_PATH}\n"
        )

        f.write(
            f"Test samples: 1267\n"
        )

        f.write(
            f"Device: {DEVICE}\n\n"
        )

        f.write(
            "OVERALL METRICS\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        for key, value in metrics.items():

            f.write(
                f"{key}: {value:.4f}\n"
            )

        f.write("\n")

        f.write(
            "CLASSIFICATION REPORT\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            classification_report_text
        )

        f.write("\n")

        f.write(
            "CONFUSION MATRIX\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            "Rows = True labels\n"
        )

        f.write(
            "Columns = Predicted labels\n\n"
        )

        f.write(
            str(cm)
        )

        f.write("\n\n")

        f.write(
            "PREDICTION DISTRIBUTION\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        for key, value in distribution.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write("\n")

        f.write(
            "CONFIDENCE ANALYSIS\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        for key, value in confidence_stats.items():

            if isinstance(value, float):

                f.write(
                    f"{key}: {value:.4f}\n"
                )

            else:

                f.write(
                    f"{key}: {value}\n"
                )

    print(
        f"Saved report: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "BERT V2 MODEL EVALUATION"
    )

    print(
        f"CUDA available: "
        f"{'YES' if torch.cuda.is_available() else 'NO'}"
    )

    print(
        f"Using device: {DEVICE}"
    )

    # --------------------------------------------------------
    # 1. LOAD TEST DATA
    # --------------------------------------------------------

    test_dataset = load_test_dataset()

    # --------------------------------------------------------
    # 2. VALIDATE DATASET
    # --------------------------------------------------------

    validate_dataset(
        test_dataset
    )

    # --------------------------------------------------------
    # 3. LOAD TOKENIZER
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # Prevent unused-variable warning
    _ = tokenizer

    # --------------------------------------------------------
    # 4. LOAD MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # 5. CREATE TRAINER
    # --------------------------------------------------------

    trainer = create_trainer(
        model
    )

    # --------------------------------------------------------
    # 6. RUN PREDICTIONS
    # --------------------------------------------------------

    (
        predicted_labels,
        probabilities,
        confidence
    ) = run_predictions(
        trainer,
        test_dataset
    )

    # --------------------------------------------------------
    # 7. TRUE LABELS
    # --------------------------------------------------------

    true_labels = np.array(
        test_dataset["labels"]
    )

    # --------------------------------------------------------
    # 8. OVERALL METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        true_labels,
        predicted_labels
    )

    # --------------------------------------------------------
    # 9. CLASSIFICATION REPORT
    # --------------------------------------------------------

    report_text = create_classification_report(
        true_labels,
        predicted_labels
    )

    # --------------------------------------------------------
    # 10. CONFUSION MATRIX
    # --------------------------------------------------------

    cm = create_confusion_matrix(
        true_labels,
        predicted_labels
    )

    # --------------------------------------------------------
    # 11. DISTRIBUTION
    # --------------------------------------------------------

    distribution = analyze_prediction_distribution(
        true_labels,
        predicted_labels
    )

    # --------------------------------------------------------
    # 12. CONFIDENCE
    # --------------------------------------------------------

    confidence_stats = analyze_confidence(
        true_labels,
        predicted_labels,
        confidence
    )

    # --------------------------------------------------------
    # 13. BUILD DATAFRAME
    # --------------------------------------------------------

    results_df = build_predictions_dataframe(
        test_dataset,
        true_labels,
        predicted_labels,
        probabilities,
        confidence
    )

    # --------------------------------------------------------
    # 14. ERROR DIRECTION
    # --------------------------------------------------------

    analyze_error_direction(
        results_df
    )

    # --------------------------------------------------------
    # 15. SAVE CSV FILES
    # --------------------------------------------------------

    save_prediction_files(
        results_df
    )

    # --------------------------------------------------------
    # 16. SAVE METRICS
    # --------------------------------------------------------

    save_metrics(
        metrics,
        confidence_stats,
        distribution
    )

    # --------------------------------------------------------
    # 17. SAVE TEXT REPORT
    # --------------------------------------------------------

    save_text_report(
        metrics,
        report_text,
        cm,
        distribution,
        confidence_stats
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_header(
        "BERT V2 EVALUATION COMPLETED"
    )

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print("\nGenerated files:")

    print(
        "  - all_predictions.csv"
    )

    print(
        "  - misclassified.csv"
    )

    print(
        "  - high_confidence_errors.csv"
    )

    print(
        "  - confusion_matrix.csv"
    )

    print(
        "  - metrics.json"
    )

    print(
        "  - evaluation_report.txt"
    )

    print(
        "  - plots/confusion_matrix.png"
    )

    print(
        "  - plots/confidence_distribution.png"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()