import pandas as pd
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

# Selected based on tokenization analysis of LIAR dataset
MAX_LENGTH = 128

INPUT_DIRS = {
    "binary": Path("data/processed/binary"),
    "three_class": Path("data/processed/three_class")
}

OUTPUT_DIR = Path("data/processed/bert")

SPLITS = [
    "train",
    "validation",
    "test"
]


# ============================================================
# LABEL MAPPINGS
# ============================================================

BINARY_LABELS = {
    "FAKE": 0,
    "REAL": 1
}


THREE_CLASS_LABELS = {
    "FAKE": 0,
    "UNCERTAIN": 1,
    "REAL": 2
}


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
    print(f"Maximum sequence length: {MAX_LENGTH}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded successfully.")

    return tokenizer


# ============================================================
# LOAD DATASET
# ============================================================

def load_csv(file_path):
    """
    Load a previously processed LIAR CSV dataset.

    The preprocessing stage has already handled:
    - missing values
    - conflicting statements
    - duplicate statements
    - label mapping
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "statement",
        "label"
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path}: "
            f"{missing_columns}"
        )

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    df,
    label_mapping,
    split_name
):
    """
    Validate the already-preprocessed dataset.

    This function DOES NOT remove duplicates.
    It only checks whether unexpected duplicates exist.
    """

    print(
        f"\nValidating {split_name} dataset..."
    )

    # --------------------------------------------------------
    # Check missing statements
    # --------------------------------------------------------

    missing_statements = (
        df["statement"].isna().sum()
    )

    if missing_statements > 0:
        raise ValueError(
            f"{split_name}: "
            f"{missing_statements} missing statements found."
        )

    # --------------------------------------------------------
    # Check missing labels
    # --------------------------------------------------------

    missing_labels = (
        df["label"].isna().sum()
    )

    if missing_labels > 0:
        raise ValueError(
            f"{split_name}: "
            f"{missing_labels} missing labels found."
        )

    # --------------------------------------------------------
    # Check empty statements
    # --------------------------------------------------------

    empty_statements = (
        df["statement"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_statements > 0:
        raise ValueError(
            f"{split_name}: "
            f"{empty_statements} empty statements found."
        )

    # --------------------------------------------------------
    # Check duplicate statements
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # We DO NOT remove duplicates here.
    #
    # Duplicates were already removed during preprocessing.
    # This is only a safety check.
    # --------------------------------------------------------

    duplicate_count = (
        df["statement"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{split_name}: "
            f"{duplicate_count} duplicate statements found. "
            f"Please review the preprocessing stage."
        )

    # --------------------------------------------------------
    # Check unexpected labels
    # --------------------------------------------------------

    actual_labels = set(
        df["label"].unique()
    )

    expected_labels = set(
        label_mapping.keys()
    )

    unexpected_labels = (
        actual_labels - expected_labels
    )

    if unexpected_labels:
        raise ValueError(
            f"{split_name}: "
            f"Unexpected labels found: "
            f"{unexpected_labels}"
        )

    # --------------------------------------------------------
    # Check label completeness
    # --------------------------------------------------------

    missing_expected_labels = (
        expected_labels - actual_labels
    )

    if missing_expected_labels:
        print(
            f"WARNING: {split_name} does not contain "
            f"the following expected labels: "
            f"{missing_expected_labels}"
        )

    print("Validation passed.")


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_dataset(
    df,
    tokenizer,
    label_mapping
):
    """
    Convert statements into BERT-compatible inputs.

    Final model inputs:
        input_ids
        attention_mask
        token_type_ids (if provided by tokenizer)
        labels
    """

    df = df.copy()

    # --------------------------------------------------------
    # Convert labels to integer IDs
    # --------------------------------------------------------

    df["labels"] = (
        df["label"]
        .map(label_mapping)
        .astype(int)
    )

    # --------------------------------------------------------
    # Keep only the columns needed for tokenization
    # --------------------------------------------------------

    df = df[
        [
            "statement",
            "labels"
        ]
    ]

    # --------------------------------------------------------
    # Convert Pandas DataFrame to Hugging Face Dataset
    # --------------------------------------------------------

    dataset = Dataset.from_pandas(
        df,
        preserve_index=False
    )

    # --------------------------------------------------------
    # Tokenization function
    # --------------------------------------------------------

    def tokenize_batch(batch):

        return tokenizer(
            batch["statement"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    # --------------------------------------------------------
    # Tokenize dataset
    # --------------------------------------------------------

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        desc="Tokenizing dataset"
    )

    # --------------------------------------------------------
    # Remove original text
    # --------------------------------------------------------
    #
    # BERT no longer needs the original statement after
    # tokenization.
    # --------------------------------------------------------

    tokenized_dataset = tokenized_dataset.remove_columns(
        ["statement"]
    )

    return tokenized_dataset


# ============================================================
# VERIFY TOKENIZED DATASET
# ============================================================

def verify_tokenized_dataset(
    dataset,
    expected_label_count
):
    """
    Verify the structure of the final BERT dataset.
    """

    print("\nVerifying tokenized dataset...")

    columns = dataset.column_names

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
        "input_ids",
        "attention_mask",
        "labels"
    }

    missing_columns = (
        required_columns - set(columns)
    )

    if missing_columns:
        raise ValueError(
            "Required BERT columns are missing: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Verify sample count
    # --------------------------------------------------------

    if len(dataset) == 0:
        raise ValueError(
            "Tokenized dataset is empty."
        )

    # --------------------------------------------------------
    # Verify sequence length
    # --------------------------------------------------------

    first_sample = dataset[0]

    input_length = len(
        first_sample["input_ids"]
    )

    attention_length = len(
        first_sample["attention_mask"]
    )

    if input_length != MAX_LENGTH:
        raise ValueError(
            f"Unexpected input length: "
            f"{input_length}. "
            f"Expected: {MAX_LENGTH}"
        )

    if attention_length != MAX_LENGTH:
        raise ValueError(
            f"Unexpected attention mask length: "
            f"{attention_length}. "
            f"Expected: {MAX_LENGTH}"
        )

    # --------------------------------------------------------
    # Verify label range
    # --------------------------------------------------------

    labels = dataset["labels"]

    invalid_labels = [
        label
        for label in labels
        if label < 0 or label >= expected_label_count
    ]

    if invalid_labels:
        raise ValueError(
            f"Invalid label IDs found: "
            f"{set(invalid_labels)}"
        )

    print("Tokenized dataset validation passed.")


# ============================================================
# SHOW SAMPLE
# ============================================================

def show_sample(
    dataset,
    tokenizer
):
    """
    Display one tokenized example for manual verification.
    """

    print("\n" + "-" * 60)
    print("TOKENIZATION SAMPLE")
    print("-" * 60)

    sample = dataset[0]

    print(
        f"\nLabel ID: "
        f"{sample['labels']}"
    )

    print(
        f"Input IDs length: "
        f"{len(sample['input_ids'])}"
    )

    print(
        f"Attention mask length: "
        f"{len(sample['attention_mask'])}"
    )

    if "token_type_ids" in sample:

        print(
            f"Token type IDs length: "
            f"{len(sample['token_type_ids'])}"
        )

    # --------------------------------------------------------
    # Convert IDs back to tokens
    # --------------------------------------------------------

    tokens = tokenizer.convert_ids_to_tokens(
        sample["input_ids"]
    )

    print("\nFirst 30 tokens:")

    print(
        tokens[:30]
    )

    # --------------------------------------------------------
    # Decode text
    # --------------------------------------------------------

    decoded_text = tokenizer.decode(
        sample["input_ids"],
        skip_special_tokens=True
    )

    print("\nDecoded text:")

    print(decoded_text)


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    dataset,
    output_path
):
    """
    Save tokenized Hugging Face Dataset to disk.
    """

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.save_to_disk(
        str(output_path)
    )

    print(
        f"\nSaved tokenized dataset to:"
        f"\n{output_path}"
    )

    print(
        f"Number of samples: "
        f"{len(dataset)}"
    )

    print(
        f"Columns: "
        f"{dataset.column_names}"
    )


# ============================================================
# PROCESS CLASSIFICATION TYPE
# ============================================================

def process_classification_type(
    classification_type,
    tokenizer,
    label_mapping
):
    """
    Process train, validation and test datasets
    for one classification strategy.
    """

    print("\n\n" + "=" * 60)

    print(
        f"{classification_type.upper()} "
        f"BERT DATASET"
    )

    print("=" * 60)

    input_dir = INPUT_DIRS[
        classification_type
    ]

    expected_label_count = len(
        label_mapping
    )

    for split_name in SPLITS:

        print("\n" + "-" * 60)

        print(
            f"PROCESSING {split_name.upper()}"
        )

        print("-" * 60)

        # ----------------------------------------------------
        # Input file
        # ----------------------------------------------------

        input_file = (
            input_dir
            / f"{split_name}.csv"
        )

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        df = load_csv(
            input_file
        )

        print(
            f"Samples before tokenization: "
            f"{len(df)}"
        )

        # ----------------------------------------------------
        # Validate dataset
        # ----------------------------------------------------

        validate_dataset(
            df=df,
            label_mapping=label_mapping,
            split_name=split_name
        )

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        tokenized_dataset = tokenize_dataset(
            df=df,
            tokenizer=tokenizer,
            label_mapping=label_mapping
        )

        # ----------------------------------------------------
        # Verify tokenized dataset
        # ----------------------------------------------------

        verify_tokenized_dataset(
            dataset=tokenized_dataset,
            expected_label_count=expected_label_count
        )

        # ----------------------------------------------------
        # Output path
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / classification_type
            / split_name
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_dataset(
            dataset=tokenized_dataset,
            output_path=output_path
        )

        # ----------------------------------------------------
        # Show sample only for training dataset
        # ----------------------------------------------------

        if split_name == "train":

            show_sample(
                dataset=tokenized_dataset,
                tokenizer=tokenizer
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LIAR DATASET - BERT PREPARATION")
    print("=" * 60)

    print(
        f"\nBERT model: {MODEL_NAME}"
    )

    print(
        f"Maximum sequence length: "
        f"{MAX_LENGTH}"
    )

    print(
        "\nClassification strategies:"
    )

    print(
        "1. Binary: FAKE / REAL"
    )

    print(
        "2. Three-class: "
        "FAKE / UNCERTAIN / REAL"
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Process Binary dataset
    # --------------------------------------------------------

    process_classification_type(
        classification_type="binary",
        tokenizer=tokenizer,
        label_mapping=BINARY_LABELS
    )

    # --------------------------------------------------------
    # Process Three-Class dataset
    # --------------------------------------------------------

    process_classification_type(
        classification_type="three_class",
        tokenizer=tokenizer,
        label_mapping=THREE_CLASS_LABELS
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n\n" + "=" * 60)
    print("BERT DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print(
        "\nTokenized datasets saved under:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nReady for BERT fine-tuning."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()