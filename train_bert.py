import numpy as np
import torch

from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

DATASET_DIR = Path("data/processed/bert/binary")

OUTPUT_DIR = Path("models/bert/binary")

FINAL_MODEL_DIR = OUTPUT_DIR / "final"

SEED = 42

NUM_LABELS = 2

NUM_EPOCHS = 3

LEARNING_RATE = 2e-5

TRAIN_BATCH_SIZE = 8

EVAL_BATCH_SIZE = 8

WEIGHT_DECAY = 0.01

MAX_LENGTH = 128


# ============================================================
# LABEL CONFIGURATION
# ============================================================

LABEL2ID = {
    "FAKE": 0,
    "REAL": 1
}

ID2LABEL = {
    0: "FAKE",
    1: "REAL"
}


# ============================================================
# DEVICE
# ============================================================

def detect_device():

    print("\n" + "=" * 60)
    print("DEVICE CONFIGURATION")
    print("=" * 60)

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("CUDA available: YES")
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print("CUDA available: NO")
        print("Using CPU for training.")

    return device


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    print("\n" + "=" * 60)
    print("LOADING BERT DATASETS")
    print("=" * 60)

    train_path = DATASET_DIR / "train"

    validation_path = DATASET_DIR / "validation"

    test_path = DATASET_DIR / "test"

    train_dataset = load_from_disk(
        str(train_path)
    )

    validation_dataset = load_from_disk(
        str(validation_path)
    )

    test_dataset = load_from_disk(
        str(test_path)
    )

    print(
        f"\nTraining samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    print("\nDataset columns:")

    print(
        train_dataset.column_names
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


# ============================================================
# VALIDATE DATASETS
# ============================================================

def validate_datasets(
    train_dataset,
    validation_dataset,
    test_dataset
):

    print("\n" + "=" * 60)
    print("VALIDATING DATASETS")
    print("=" * 60)

    required_columns = {
        "labels",
        "input_ids",
        "attention_mask"
    }

    for name, dataset in [
        ("train", train_dataset),
        ("validation", validation_dataset),
        ("test", test_dataset)
    ]:

        columns = set(
            dataset.column_names
        )

        missing_columns = (
            required_columns - columns
        )

        if missing_columns:

            raise ValueError(
                f"{name} dataset is missing "
                f"columns: {missing_columns}"
            )

        labels = dataset["labels"]

        invalid_labels = [
            label
            for label in labels
            if label not in ID2LABEL
        ]

        if invalid_labels:

            raise ValueError(
                f"Invalid labels found in "
                f"{name}: {set(invalid_labels)}"
            )

        for input_ids, attention_mask in zip(
            dataset["input_ids"],
            dataset["attention_mask"]
        ):

            if len(input_ids) != MAX_LENGTH:

                raise ValueError(
                    f"{name}: input_ids length "
                    f"is {len(input_ids)}, "
                    f"expected {MAX_LENGTH}"
                )

            if len(attention_mask) != MAX_LENGTH:

                raise ValueError(
                    f"{name}: attention_mask length "
                    f"is {len(attention_mask)}, "
                    f"expected {MAX_LENGTH}"
                )

    print("Dataset validation passed.")


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    print("\n" + "=" * 60)
    print("LOADING BERT TOKENIZER")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"Maximum sequence length: "
        f"{MAX_LENGTH}"
    )

    print("Tokenizer loaded successfully.")

    return tokenizer


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 60)
    print("LOADING BERT MODEL")
    print("=" * 60)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    print(
        f"Number of labels: "
        f"{NUM_LABELS}"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    print(
        "\nBERT sequence classification "
        "model loaded successfully."
    )

    return model


# ============================================================
# COMPUTE METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    # Predictions are logits
    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# CREATE TRAINING ARGUMENTS
# ============================================================

def create_training_arguments():

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    training_args = TrainingArguments(

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_dir=str(
            OUTPUT_DIR
        ),

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        num_train_epochs=NUM_EPOCHS,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        weight_decay=WEIGHT_DECAY,

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optim="adamw_torch",

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        eval_strategy="epoch",

        # ----------------------------------------------------
        # Checkpoints
        # ----------------------------------------------------

        save_strategy="epoch",

        save_total_limit=2,

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logging_strategy="steps",

        logging_steps=100,

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        seed=SEED,

        # ----------------------------------------------------
        # CPU / GPU
        # ----------------------------------------------------

        use_cpu=not torch.cuda.is_available(),

        fp16=False,

        # ----------------------------------------------------
        # Reporting
        # ----------------------------------------------------

        report_to="none"
    )

    print(
        f"\nEpochs: {NUM_EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Training batch size: "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        f"Evaluation batch size: "
        f"{EVAL_BATCH_SIZE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        "Evaluation strategy: epoch"
    )

    print(
        "Best model metric: F1"
    )

    print(
        f"FP16: {False}"
    )

    print(
        f"Device: "
        f"{'CUDA' if torch.cuda.is_available() else 'CPU'}"
    )

    return training_args


# ============================================================
# CREATE TRAINER
# ============================================================

def create_trainer(
    model,
    training_args,
    train_dataset,
    validation_dataset,
    tokenizer
):

    print("\n" + "=" * 60)
    print("CREATING TRAINER")
    print("=" * 60)

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        compute_metrics=compute_metrics
    )

    print(
        "Trainer created successfully."
    )

    return trainer


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(trainer):

    print("\n" + "=" * 60)
    print("STARTING BERT FINE-TUNING")
    print("=" * 60)

    print(
        "\nTraining will now begin..."
    )

    print(
        "This may take a while because "
        "the current environment uses CPU."
    )

    train_result = trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("\nTraining metrics:")

    print(
        train_result.metrics
    )

    return train_result


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    trainer,
    test_dataset
):

    print("\n" + "=" * 60)
    print("EVALUATING BERT ON TEST DATA")
    print("=" * 60)

    test_results = trainer.evaluate(
        test_dataset
    )

    print("\nTest metrics:")

    for key, value in test_results.items():

        if isinstance(value, float):

            print(
                f"{key}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    return test_results


# ============================================================
# CONFUSION MATRIX
# ============================================================

def generate_confusion_matrix(
    trainer,
    test_dataset
):

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    predictions = trainer.predict(
        test_dataset
    )

    predicted_labels = np.argmax(
        predictions.predictions,
        axis=1
    )

    true_labels = (
        predictions.label_ids
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels
    )

    print("\nRows = True labels")
    print("Columns = Predicted labels")

    print("\nLabels:")

    print(
        "[FAKE, REAL]"
    )

    print("\nConfusion matrix:")

    print(matrix)

    return matrix


# ============================================================
# SAVE MODEL
# ============================================================

def save_final_model(
    trainer,
    tokenizer
):

    print("\n" + "=" * 60)
    print("SAVING FINAL BERT MODEL")
    print("=" * 60)

    FINAL_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trainer.save_model(
        str(FINAL_MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(FINAL_MODEL_DIR)
    )

    print(
        "\nFinal model saved to:"
    )

    print(
        FINAL_MODEL_DIR
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BERT FINE-TUNING")
    print("=" * 60)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Detect device
    # --------------------------------------------------------

    detect_device()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        test_dataset
    ) = load_datasets()

    # --------------------------------------------------------
    # Validate datasets
    # --------------------------------------------------------

    validate_datasets(
        train_dataset,
        validation_dataset,
        test_dataset
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Load BERT
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = (
        create_training_arguments()
    )

    # --------------------------------------------------------
    # Create trainer
    # --------------------------------------------------------

    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        tokenizer=tokenizer
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_model(
        trainer
    )

    # --------------------------------------------------------
    # Evaluate on test set
    # --------------------------------------------------------

    evaluate_model(
        trainer,
        test_dataset
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    generate_confusion_matrix(
        trainer,
        test_dataset
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    save_final_model(
        trainer,
        tokenizer
    )

    print("\n" + "=" * 60)
    print("BERT FINE-TUNING PIPELINE COMPLETED")
    print("=" * 60)

    print(
        "\nModel ready for inference and comparison."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()