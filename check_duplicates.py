import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "BINARY": Path("data/processed/binary/train.csv"),
    "THREE-CLASS": Path("data/processed/three_class/train.csv")
}


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(dataset_name, file_path):

    print("\n" + "=" * 60)
    print(f"{dataset_name} DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(file_path)

    print(f"\nTotal samples: {len(df)}")

    # --------------------------------------------------------
    # Find duplicate statements
    # --------------------------------------------------------

    duplicates = df[
        df["statement"].duplicated(keep=False)
    ].copy()

    unique_duplicate_statements = (
        duplicates["statement"].nunique()
    )

    print(f"Duplicate rows: {len(duplicates)}")
    print(
        f"Unique duplicated statements: "
        f"{unique_duplicate_statements}"
    )

    # --------------------------------------------------------
    # Analyze labels
    # --------------------------------------------------------

    conflicting_duplicates = []

    print("\n" + "-" * 60)
    print("DUPLICATE STATEMENTS")
    print("-" * 60)

    for statement, group in duplicates.groupby("statement"):

        labels = group["label"].unique()

        print("\nStatement:")
        print(statement)

        print("Labels:")
        print(list(labels))

        # Check whether the same statement
        # has different labels
        if len(labels) > 1:

            conflicting_duplicates.append({
                "statement": statement,
                "labels": list(labels)
            })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)

    if len(conflicting_duplicates) == 0:

        print("\n✓ No conflicting labels found.")

        print(
            "✓ All duplicated statements have "
            "the same label."
        )

    else:

        print(
            f"\n⚠ Conflicting labels found: "
            f"{len(conflicting_duplicates)}"
        )

        for item in conflicting_duplicates:

            print("\nConflicting statement:")
            print(item["statement"])

            print("Labels:")
            print(item["labels"])

    return {
        "dataset": dataset_name,
        "total_samples": len(df),
        "duplicate_rows": len(duplicates),
        "unique_duplicates": unique_duplicate_statements,
        "conflicting_duplicates": len(conflicting_duplicates)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LIAR DATASET - DUPLICATE ANALYSIS")
    print("=" * 60)

    results = []

    for dataset_name, file_path in DATASETS.items():

        if not file_path.exists():

            print(
                f"\nERROR: File not found: {file_path}"
            )

            continue

        result = analyze_duplicates(
            dataset_name,
            file_path
        )

        results.append(result)

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    if results:

        summary_df = pd.DataFrame(results)

        print(summary_df.to_string(index=False))

    print("\nAnalysis completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()