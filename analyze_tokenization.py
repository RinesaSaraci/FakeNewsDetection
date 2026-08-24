import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

DATASETS = {
    "BINARY": Path("data/processed/binary"),
    "THREE_CLASS": Path("data/processed/three_class")
}

SPLITS = [
    "train",
    "validation",
    "test"
]


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():
    """
    Load the pretrained BERT tokenizer.
    """

    print("=" * 60)
    print("LOADING BERT TOKENIZER")
    print("=" * 60)

    print(f"\nModel: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded successfully.")

    return tokenizer


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """
    Load a processed LIAR CSV file.
    """

    return pd.read_csv(file_path)


# ============================================================
# TOKENIZATION ANALYSIS
# ============================================================

def analyze_token_lengths(df, tokenizer):
    """
    Tokenize all statements and calculate token lengths.

    Special tokens [CLS] and [SEP] are included because
    they are part of the actual BERT input.
    """

    statements = df["statement"].tolist()

    token_lengths = []

    for statement in statements:

        encoding = tokenizer(
            statement,
            add_special_tokens=True,
            truncation=False
        )

        token_lengths.append(
            len(encoding["input_ids"])
        )

    return pd.Series(token_lengths)


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(token_lengths):

    statistics = {
        "count": len(token_lengths),
        "minimum": token_lengths.min(),
        "maximum": token_lengths.max(),
        "mean": token_lengths.mean(),
        "median": token_lengths.median(),
        "90th_percentile": token_lengths.quantile(0.90),
        "95th_percentile": token_lengths.quantile(0.95),
        "99th_percentile": token_lengths.quantile(0.99)
    }

    return statistics


# ============================================================
# TRUNCATION ANALYSIS
# ============================================================

def analyze_truncation(token_lengths):

    max_lengths = [
        64,
        128,
        256,
        512
    ]

    results = []

    for max_length in max_lengths:

        truncated = (
            token_lengths > max_length
        ).sum()

        percentage = (
            truncated / len(token_lengths)
        ) * 100

        results.append({
            "max_length": max_length,
            "truncated_samples": truncated,
            "percentage": percentage
        })

    return results


# ============================================================
# PRINT ANALYSIS
# ============================================================

def print_analysis(
    dataset_name,
    split_name,
    statistics,
    truncation_results
):

    print("\n" + "=" * 60)
    print(f"{dataset_name} - {split_name.upper()}")
    print("=" * 60)

    print("\nToken length statistics:")

    print(
        f"Samples:              "
        f"{statistics['count']}"
    )

    print(
        f"Minimum:              "
        f"{statistics['minimum']}"
    )

    print(
        f"Maximum:              "
        f"{statistics['maximum']}"
    )

    print(
        f"Average:              "
        f"{statistics['mean']:.2f}"
    )

    print(
        f"Median:               "
        f"{statistics['median']:.2f}"
    )

    print(
        f"90th percentile:      "
        f"{statistics['90th_percentile']:.2f}"
    )

    print(
        f"95th percentile:      "
        f"{statistics['95th_percentile']:.2f}"
    )

    print(
        f"99th percentile:      "
        f"{statistics['99th_percentile']:.2f}"
    )

    print("\nTruncation analysis:")

    print(
        f"{'max_length':<15}"
        f"{'truncated':<15}"
        f"{'percentage':<15}"
    )

    print("-" * 45)

    for result in truncation_results:

        print(
            f"{result['max_length']:<15}"
            f"{result['truncated_samples']:<15}"
            f"{result['percentage']:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LIAR DATASET - BERT TOKENIZATION ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Show tokenizer information
    # --------------------------------------------------------

    print("\nTokenizer information:")

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"Model max length: "
        f"{tokenizer.model_max_length}"
    )

    print(
        f"CLS token: "
        f"{tokenizer.cls_token}"
    )

    print(
        f"SEP token: "
        f"{tokenizer.sep_token}"
    )

    # --------------------------------------------------------
    # Analyze datasets
    # --------------------------------------------------------

    for dataset_name, dataset_dir in DATASETS.items():

        for split_name in SPLITS:

            file_path = (
                dataset_dir
                / f"{split_name}.csv"
            )

            print(
                f"\nProcessing: "
                f"{file_path}"
            )

            if not file_path.exists():

                print(
                    f"ERROR: File not found: "
                    f"{file_path}"
                )

                continue

            # Load data
            df = load_dataset(file_path)

            # Tokenization analysis
            token_lengths = analyze_token_lengths(
                df,
                tokenizer
            )

            # Statistics
            statistics = calculate_statistics(
                token_lengths
            )

            # Truncation
            truncation_results = analyze_truncation(
                token_lengths
            )

            # Print results
            print_analysis(
                dataset_name,
                split_name,
                statistics,
                truncation_results
            )

    print("\n\n" + "=" * 60)
    print("TOKENIZATION ANALYSIS COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()