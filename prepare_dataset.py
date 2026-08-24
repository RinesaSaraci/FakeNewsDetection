import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATA_DIR = Path("data/raw/liar_dataset")
PROCESSED_DATA_DIR = Path("data/processed")

LABEL_COLUMN = 1
STATEMENT_COLUMN = 2


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


THREE_CLASS_MAPPING = {
    "pants-fire": "FAKE",
    "false": "FAKE",
    "barely-true": "UNCERTAIN",
    "half-true": "UNCERTAIN",
    "mostly-true": "REAL",
    "true": "REAL"
}


# ============================================================
# DATASET FILES
# ============================================================

FILES = {
    "train": RAW_DATA_DIR / "train.tsv",
    "validation": RAW_DATA_DIR / "valid.tsv",
    "test": RAW_DATA_DIR / "test.tsv"
}


# ============================================================
# LOAD DATA
# ============================================================

def load_liar_dataset(file_path):
    """Load a LIAR TSV file."""

    return pd.read_csv(
        file_path,
        sep="\t",
        header=None
    )


# ============================================================
# PREPARE BASIC DATASET
# ============================================================

def prepare_dataset(df, label_mapping):
    """
    Select the statement and original label,
    map the original labels to the target classes,
    and perform basic cleaning.
    """

    result = pd.DataFrame({
        "statement": df[STATEMENT_COLUMN],
        "label": df[LABEL_COLUMN]
    })

    # --------------------------------------------------------
    # Map original LIAR labels
    # --------------------------------------------------------

    result["label"] = result["label"].map(label_mapping)

    # Remove rows with labels that could not be mapped
    result = result.dropna(subset=["label"])

    # Remove missing statements
    result = result.dropna(subset=["statement"])

    # Convert statements to strings
    result["statement"] = result["statement"].astype(str)

    # Remove leading/trailing whitespace
    result["statement"] = result["statement"].str.strip()

    # Remove empty statements
    result = result[result["statement"] != ""]

    return result


# ============================================================
# HANDLE DUPLICATES
# ============================================================

def clean_training_duplicates(df):
    """
    Handle duplicate statements in the training dataset.

    Rules:
    1. Statements with conflicting labels are removed.
    2. Duplicate statements with the same label are reduced
       to one occurrence.
    """

    original_size = len(df)

    # --------------------------------------------------------
    # Find statements with conflicting labels
    # --------------------------------------------------------

    label_counts = (
        df.groupby("statement")["label"]
        .nunique()
    )

    conflicting_statements = label_counts[
        label_counts > 1
    ].index

    conflicting_count = len(conflicting_statements)

    # --------------------------------------------------------
    # Remove conflicting statements
    # --------------------------------------------------------

    if conflicting_count > 0:

        df = df[
            ~df["statement"].isin(conflicting_statements)
        ].copy()

    after_conflicts = len(df)

    # --------------------------------------------------------
    # Remove exact duplicate statements
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["statement"],
        keep="first"
    ).reset_index(drop=True)

    duplicate_rows_removed = (
        after_conflicts - len(df)
    )

    total_rows_removed = (
        original_size - len(df)
    )

    return (
        df,
        conflicting_count,
        duplicate_rows_removed,
        total_rows_removed
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df, output_path):
    """Save processed dataset as CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")
    print(f"Samples: {len(df)}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LIAR DATASET PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Process both classification strategies
    # --------------------------------------------------------

    mappings = {
        "binary": BINARY_MAPPING,
        "three_class": THREE_CLASS_MAPPING
    }

    for classification_type, mapping in mappings.items():

        print(f"\n\n{'=' * 60}")
        print(f"{classification_type.upper()} CLASSIFICATION")
        print(f"{'=' * 60}")

        for split_name, file_path in FILES.items():

            print(f"\nProcessing {split_name}...")

            # ------------------------------------------------
            # Check file
            # ------------------------------------------------

            if not file_path.exists():

                print(
                    f"ERROR: File not found: {file_path}"
                )

                continue

            # ------------------------------------------------
            # Load raw dataset
            # ------------------------------------------------

            df = load_liar_dataset(file_path)

            original_size = len(df)

            # ------------------------------------------------
            # Basic preprocessing
            # ------------------------------------------------

            processed_df = prepare_dataset(
                df,
                mapping
            )

            # ------------------------------------------------
            # Clean training duplicates
            # ------------------------------------------------

            if split_name == "train":

                (
                    processed_df,
                    conflicting_count,
                    duplicate_rows_removed,
                    total_rows_removed
                ) = clean_training_duplicates(
                    processed_df
                )

                print("\nTraining data cleaning:")

                print(
                    f"Original samples: "
                    f"{original_size}"
                )

                print(
                    f"Conflicting statements removed: "
                    f"{conflicting_count}"
                )

                print(
                    f"Duplicate rows removed: "
                    f"{duplicate_rows_removed}"
                )

                print(
                    f"Total samples removed: "
                    f"{total_rows_removed}"
                )

                print(
                    f"Final samples: "
                    f"{len(processed_df)}"
                )

            # ------------------------------------------------
            # Output path
            # ------------------------------------------------

            output_path = (
                PROCESSED_DATA_DIR
                / classification_type
                / f"{split_name}.csv"
            )

            # ------------------------------------------------
            # Save processed dataset
            # ------------------------------------------------

            save_dataset(
                processed_df,
                output_path
            )

            # ------------------------------------------------
            # Show label distribution
            # ------------------------------------------------

            print("\nLabel distribution:")

            print(
                processed_df["label"]
                .value_counts()
            )

    print("\n\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()