import os
import random
import numpy as np
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

TRAIN_PATH = r"data\processed\bert\binary\train"
VALIDATION_PATH = r"data\processed\bert\binary\validation"
TEST_PATH = r"data\processed\bert\binary\test"

OUTPUT_DIR = r"models\bert\binary_v2"
CHECKPOINT_DIR = r"results\bert\binary_v2"

SEED = 42

NUM_LABELS = 2

LABEL2ID = {
    "FAKE": 0,
    "REAL": 1,
}

ID2LABEL = {
    0: "FAKE",
    1: "REAL",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def setup_device():

    print("=" * 60)
    print("DEVICE CONFIGURATION")
    print("=" * 60)

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("CUDA available: YES")
        print("Using GPU:", torch.cuda.get_device_name(0))

    else:

        device = torch.device("cpu")

        print("CUDA available: NO")
        print("Using CPU for training.")

    print()

    return device


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    print("=" * 60)
    print("LOADING BERT DATASETS")
    print("=" * 60)

    train_dataset = load_from_disk(TRAIN_PATH)
    validation_dataset = load_from_disk(VALIDATION_PATH)
    test_dataset = load_from_disk(TEST_PATH)

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    print("\nDataset columns:")
    print(train_dataset.column_names)

    print()

    return train_dataset, validation_dataset, test_dataset


# ============================================================
# VALIDATE DATASETS
# ============================================================

def validate_datasets(train_dataset, validation_dataset, test_dataset):

    print("=" * 60)
    print("VALIDATING DATASETS")
    print("=" * 60)

    required_columns = [
        "labels",
        "input_ids",
        "token_type_ids",
        "attention_mask",
    ]

    for dataset_name, dataset in [
        ("Train", train_dataset),
        ("Validation", validation_dataset),
        ("Test", test_dataset),
    ]:

        for column in required_columns:

            if column not in dataset.column_names:

                raise ValueError(
                    f"{dataset_name} dataset is missing column: {column}"
                )

    print("Dataset validation passed.")
    print()


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    print("=" * 60)
    print("LOADING BERT TOKENIZER")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Maximum sequence length: 128")
    print("Tokenizer loaded successfully.")
    print()

    return tokenizer


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("LOADING BERT MODEL")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Number of labels: {NUM_LABELS}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )

    print("BERT sequence classification model loaded successfully.")
    print()

    return model


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_dataset):

    print("=" * 60)
    print("CALCULATING CLASS WEIGHTS")
    print("=" * 60)

    labels = np.array(train_dataset["labels"])

    class_counts = np.bincount(
        labels,
        minlength=NUM_LABELS
    )

    total = len(labels)

    weights = total / (
        NUM_LABELS * class_counts
    )

    print(f"FAKE samples: {class_counts[0]}")
    print(f"REAL samples: {class_counts[1]}")

    print(f"\nFAKE weight: {weights[0]:.4f}")
    print(f"REAL weight: {weights[1]:.4f}")

    print()

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):

    def __init__(self, class_weights=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        class_weights = self.class_weights.to(
            logits.device
        )

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights
        )

        loss = loss_function(
            logits.view(-1, NUM_LABELS),
            labels.view(-1)
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predicted_labels
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted_labels,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

def create_training_arguments():

    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    training_args = TrainingArguments(
    output_dir="models/bert/binary/v2",

    num_train_epochs=3,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    learning_rate=2e-5,
    weight_decay=0.01,

    warmup_steps=383,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    logging_strategy="steps",
    logging_steps=100,

    fp16=False,
    use_cpu=True,

    report_to="none",

    save_total_limit=2,
)

    print("Epochs: 4")
    print("Learning rate: 2e-5")
    print("Train batch size: 8")
    print("Gradient accumulation: 2")
    print("Effective batch size: 16")
    print("Evaluation batch size: 8")
    print("Weight decay: 0.01")
    print("Warmup ratio: 0.1")
    print("Dropout: 0.2")
    print("Class weighting: ENABLED")
    print("Evaluation strategy: epoch")
    print("Early stopping: ENABLED")
    print("Best model metric: F1")
    print("FP16: False")
    print("Device: CPU")

    print()

    return training_args


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    train_dataset,
    validation_dataset,
    tokenizer,
    class_weights,
    training_args,
):

    print("=" * 60)
    print("CREATING TRAINER")
    print("=" * 60)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    trainer = WeightedTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        class_weights=class_weights,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1
            )
        ],
    )

    print("Trainer created successfully.")
    print()

    print("=" * 60)
    print("STARTING BERT V2 FINE-TUNING")
    print("=" * 60)

    print()
    print("Training will now begin...")
    print()

    train_result = trainer.train()

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("\nTraining metrics:")

    print(train_result.metrics)

    return trainer


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(trainer, test_dataset):

    print()
    print("=" * 60)
    print("EVALUATING BERT V2 ON TEST DATA")
    print("=" * 60)

    test_results = trainer.evaluate(
        test_dataset
    )

    print("\nTest metrics:")

    for key, value in test_results.items():

        if isinstance(value, float):

            print(f"{key}: {value:.4f}")

        else:

            print(f"{key}: {value}")

    return test_results


# ============================================================
# DETAILED TEST REPORT
# ============================================================

def detailed_test_report(trainer, test_dataset):

    print()
    print("=" * 60)
    print("DETAILED TEST REPORT")
    print("=" * 60)

    predictions = trainer.predict(
        test_dataset
    )

    logits = predictions.predictions

    true_labels = predictions.label_ids

    predicted_labels = np.argmax(
        logits,
        axis=1
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=[
                "FAKE",
                "REAL",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    print(cm)

    print("\nRows = True labels")
    print("Columns = Predicted labels")
    print()

    print("             FAKE   REAL")
    print(f"FAKE         {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"REAL         {cm[1][0]:4d}   {cm[1][1]:4d}")

    return predictions


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(trainer, tokenizer):

    print()
    print("=" * 60)
    print("SAVING FINAL BERT V2 MODEL")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    trainer.save_model(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print()
    print("Final BERT V2 model saved to:")
    print(OUTPUT_DIR)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BERT V2 IMPROVED FINE-TUNING")
    print("=" * 60)

    print()

    set_seed(SEED)

    setup_device()

    train_dataset, validation_dataset, test_dataset = (
        load_datasets()
    )

    validate_datasets(
        train_dataset,
        validation_dataset,
        test_dataset,
    )

    tokenizer = load_tokenizer()

    model = load_model()

    class_weights = calculate_class_weights(
        train_dataset
    )

    training_args = create_training_arguments()

    trainer = train_model(
        model,
        train_dataset,
        validation_dataset,
        tokenizer,
        class_weights,
        training_args,
    )

    evaluate_test(
        trainer,
        test_dataset
    )

    detailed_test_report(
        trainer,
        test_dataset
    )

    save_model(
        trainer,
        tokenizer
    )

    print()
    print("=" * 60)
    print("BERT V2 PIPELINE COMPLETED")
    print("=" * 60)

    print()
    print("Model ready for:")
    print("1. Error analysis")
    print("2. Comparison with BERT baseline")
    print("3. Comparison with LLaMA 3")


if __name__ == "__main__":
    main()