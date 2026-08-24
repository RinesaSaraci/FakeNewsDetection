import pandas as pd
from pathlib import Path


PROCESSED_DIR = Path("data/processed")

DATASETS = {
    "binary": {
        "train": PROCESSED_DIR / "binary/train.csv",
        "validation": PROCESSED_DIR / "binary/validation.csv",
        "test": PROCESSED_DIR / "binary/test.csv",
    },
    "three_class": {
        "train": PROCESSED_DIR / "three_class/train.csv",
        "validation": PROCESSED_DIR / "three_class/validation.csv",
        "test": PROCESSED_DIR / "three_class/test.csv",
    }
}


def validate_dataset(name, file_path, expected_labels):
    print("\n" + "=" * 60)
    print(f"{name.upper()}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"Samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Empty statements
    empty_statements = (
        df["statement"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"\nEmpty statements: {empty_statements}")

    # Duplicate statements
    duplicate_statements = df["statement"].duplicated().sum()

    print(f"Duplicate statements: {duplicate_statements}")

    # Labels
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # Unexpected labels
    actual_labels = set(df["label"].unique())
    unexpected_labels = actual_labels - set(expected_labels)

    print(f"\nUnexpected labels: {unexpected_labels}")

    # Statement length
    statement_lengths = df["statement"].astype(str).str.len()

    print("\nStatement length:")
    print(f"Minimum: {statement_lengths.min()}")
    print(f"Maximum: {statement_lengths.max()}")
    print(f"Average: {statement_lengths.mean():.2f}")

    print("\nValidation completed.")


def main():

    print("=" * 60)
    print("PROCESSED LIAR DATASET VALIDATION")
    print("=" * 60)

    expected_labels = {
        "binary": ["FAKE", "REAL"],
        "three_class": ["FAKE", "UNCERTAIN", "REAL"]
    }

    for dataset_type, splits in DATASETS.items():

        print(f"\n\n{'#' * 60}")
        print(f"{dataset_type.upper()} DATASET")
        print(f"{'#' * 60}")

        for split_name, file_path in splits.items():

            validate_dataset(
                f"{dataset_type} - {split_name}",
                file_path,
                expected_labels[dataset_type]
            )


if __name__ == "__main__":
    main()