import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
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

MODEL_PATH = Path("models/bert/binary/final")

TEST_DATA_PATH = Path(
    "data/processed/bert/binary/test"
)

RAW_TEST_PATH = Path(
    "data/processed/binary/test.csv"
)

MODEL_NAME = "bert-base-uncased"

LABEL_NAMES = ["FAKE", "REAL"]

OUTPUT_DIR = Path("results/bert/binary")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_test_dataset():

    print("=" * 60)
    print("LOADING TEST DATASET")
    print("=" * 60)

    dataset = load_from_disk(
        str(TEST_DATA_PATH)
    )

    print(f"Test samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    return dataset


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 60)
    print("LOADING TRAINED BERT MODEL")
    print("=" * 60)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    print(
        f"Number of labels: "
        f"{model.config.num_labels}"
    )

    print(
        f"Label mapping: "
        f"{model.config.id2label}"
    )

    return model, tokenizer


# ============================================================
# METRICS
# ============================================================

def compute_metrics(predictions, labels):

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predicted_labels
    )

    precision = precision_score(
        labels,
        predicted_labels,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predicted_labels,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predicted_labels,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    return (
        predicted_labels,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("BERT MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    test_dataset = load_test_dataset()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, tokenizer = load_model()

    # --------------------------------------------------------
    # Create trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RUNNING PREDICTIONS")
    print("=" * 60)

    predictions = trainer.predict(
        test_dataset
    )

    logits = predictions.predictions

    labels = predictions.label_ids

    predicted_labels = np.argmax(
        logits,
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predicted_labels
    )

    precision = precision_score(
        labels,
        predicted_labels,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predicted_labels,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predicted_labels,
        pos_label=1,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(
        labels,
        predicted_labels,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0
    )

    print(report)

    # Save report
    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(
        labels,
        predicted_labels
    )

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABEL_NAMES
    )

    display.plot()

    plt.title(
        "BERT - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION")
    print("=" * 60)

    true_counts = np.bincount(
        labels,
        minlength=2
    )

    predicted_counts = np.bincount(
        predicted_labels,
        minlength=2
    )

    print("\nTrue labels:")

    for i, name in enumerate(LABEL_NAMES):
        print(
            f"{name}: {true_counts[i]}"
        )

    print("\nPredicted labels:")

    for i, name in enumerate(LABEL_NAMES):
        print(
            f"{name}: {predicted_counts[i]}"
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "logits.npy",
        logits
    )

    np.save(
        OUTPUT_DIR / "true_labels.npy",
        labels
    )

    np.save(
        OUTPUT_DIR / "predicted_labels.npy",
        predicted_labels
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()