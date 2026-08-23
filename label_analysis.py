import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw/liar_dataset")

FILES = {
    "Train": DATA_DIR / "train.tsv",
    "Validation": DATA_DIR / "valid.tsv",
    "Test": DATA_DIR / "test.tsv"
}

LABEL_COLUMN = 1


# ============================================================
# LABEL MAPPINGS
# ============================================================

# Experiment 1: Binary classification
#
# FAKE:
#   pants-fire
#   false
#   barely-true
#   half-true
#
# REAL:
#   mostly-true
#   true

BINARY_MAPPING = {
    "pants-fire": "FAKE",
    "false": "FAKE",
    "barely-true": "FAKE",
    "half-true": "FAKE",
    "mostly-true": "REAL",
    "true": "REAL"
}


# Experiment 2: Three-class classification
#
# FAKE:
#   pants-fire
#   false
#
# UNCERTAIN:
#   barely-true
#   half-true
#
# REAL:
#   mostly-true
#   true

THREE_CLASS_MAPPING = {
    "pants-fire": "FAKE",
    "false": "FAKE",
    "barely-true": "UNCERTAIN",
    "half-true": "UNCERTAIN",
    "mostly-true": "REAL",
    "true": "REAL"
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """
    Loads one LIAR TSV file.
    The dataset does not contain column names, so we use header=None.
    """
    return pd.read_csv(
        file_path,
        sep="\t",
        header=None
    )


# ============================================================
# ANALYZE LABEL DISTRIBUTION
# ============================================================

def analyze_mapping(df, mapping, mapping_name):
    """
    Applies a label mapping and prints the resulting distribution.
    """

    original_labels = df[LABEL_COLUMN]

    mapped_labels = original_labels.map(mapping)

    counts = mapped_labels.value_counts()

    total = len(mapped_labels)

    print(f"\n===== {mapping_name} =====")

    for label in ["FAKE", "UNCERTAIN", "REAL"]:
        if label in counts:
            count = counts[label]
            percentage = (count / total) * 100

            print(
                f"{label:<10} {count:>5} "
                f"({percentage:>6.2f}%)"
            )

    print(f"{'TOTAL':<10} {total:>5}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LIAR DATASET - LABEL ANALYSIS")
    print("=" * 60)

    for split_name, file_path in FILES.items():

        print(f"\n\n{'=' * 60}")
        print(f"{split_name.upper()} DATASET")
        print(f"{'=' * 60}")

        # Check if file exists
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            continue

        # Load dataset
        df = load_dataset(file_path)

        print(f"Total samples: {len(df)}")

        # ----------------------------------------------------
        # Original labels
        # ----------------------------------------------------

        print("\nOriginal label distribution:")

        original_counts = df[LABEL_COLUMN].value_counts()

        print(original_counts)

        # ----------------------------------------------------
        # Binary classification
        # ----------------------------------------------------

        analyze_mapping(
            df,
            BINARY_MAPPING,
            "BINARY CLASSIFICATION"
        )

        # ----------------------------------------------------
        # Three-class classification
        # ----------------------------------------------------

        analyze_mapping(
            df,
            THREE_CLASS_MAPPING,
            "THREE-CLASS CLASSIFICATION"
        )


if __name__ == "__main__":
    main()