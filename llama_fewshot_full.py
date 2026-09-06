import os
import re
import time
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# LLAMA 3.2 3B - OPENROUTER FEW-SHOT FULL TEST
# ============================================================

print("=" * 70)
print("LLAMA 3.2 3B - OPENROUTER FEW-SHOT FULL TEST")
print("=" * 70)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL = "meta-llama/llama-3.2-3b-instruct"

TRAIN_PATH = "data/processed/binary/train.csv"
TEST_PATH = "data/processed/binary/test.csv"

OUTPUT_PATH = (
    "results/llama/few-shot/"
    "llama_openrouter_fewshot_full_test.csv"
)

EXAMPLES_OUTPUT_PATH = (
    "results/llama/few-shot/"
    "llama_openrouter_fewshot_examples.csv"
)

REQUEST_DELAY = 0.2

RANDOM_STATE = 42

# Number of examples from each class
N_FAKE = 3
N_REAL = 3


# ============================================================
# 2. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("OPEN_ROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "[ERROR] OPEN_ROUTER_API_KEY not found in environment."
    )

print("[OK] OPEN_ROUTER_API_KEY found")


# ============================================================
# 3. CREATE OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

print("[OK] OpenRouter client created")


# ============================================================
# 4. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(EXAMPLES_OUTPUT_PATH),
    exist_ok=True
)

print("[OK] Output directories ready")


# ============================================================
# 5. LOAD TRAIN DATASET
# ============================================================

try:

    train_df = pd.read_csv(TRAIN_PATH)

except FileNotFoundError:

    raise FileNotFoundError(
        f"[ERROR] TRAIN dataset not found:\n{TRAIN_PATH}"
    )

print()
print("[OK] TRAIN dataset loaded")
print(f"[INFO] TRAIN samples: {len(train_df)}")
print(f"[INFO] TRAIN columns: {list(train_df.columns)}")


# ============================================================
# 6. LOAD TEST DATASET
# ============================================================

try:

    test_df = pd.read_csv(TEST_PATH)

except FileNotFoundError:

    raise FileNotFoundError(
        f"[ERROR] TEST dataset not found:\n{TEST_PATH}"
    )

print()
print("[OK] TEST dataset loaded")
print(f"[INFO] TEST samples: {len(test_df)}")
print(f"[INFO] TEST columns: {list(test_df.columns)}")


# ============================================================
# 7. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    "statement",
    "label"
}

if not required_columns.issubset(train_df.columns):

    raise ValueError(
        "[ERROR] TRAIN dataset must contain "
        "'statement' and 'label' columns."
    )

if not required_columns.issubset(test_df.columns):

    raise ValueError(
        "[ERROR] TEST dataset must contain "
        "'statement' and 'label' columns."
    )

print()
print("[OK] Required columns found in both datasets")


# ============================================================
# 8. REMOVE MISSING VALUES
# ============================================================

train_df = train_df.dropna(
    subset=["statement", "label"]
).copy()

test_df = test_df.dropna(
    subset=["statement", "label"]
).copy()


# ============================================================
# 9. NORMALIZE LABELS
# ============================================================

train_df["label"] = (
    train_df["label"]
    .astype(str)
    .str.upper()
    .str.strip()
)

test_df["label"] = (
    test_df["label"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ============================================================
# 10. VALIDATE BINARY LABELS
# ============================================================

valid_labels = {
    "FAKE",
    "REAL"
}

train_labels = set(
    train_df["label"].unique()
)

test_labels = set(
    test_df["label"].unique()
)

if not train_labels.issubset(valid_labels):

    invalid = train_labels - valid_labels

    raise ValueError(
        f"[ERROR] Invalid TRAIN labels: {invalid}"
    )

if not test_labels.issubset(valid_labels):

    invalid = test_labels - valid_labels

    raise ValueError(
        f"[ERROR] Invalid TEST labels: {invalid}"
    )

print()
print("[OK] Binary FAKE/REAL labels confirmed")


# ============================================================
# 11. DISPLAY DATASET DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("DATASET DISTRIBUTION")
print("=" * 70)

print()
print("TRAIN:")
print(train_df["label"].value_counts())

print()
print("TEST:")
print(test_df["label"].value_counts())


# ============================================================
# 12. CHECK ENOUGH FEW-SHOT EXAMPLES
# ============================================================

fake_count = (
    train_df["label"] == "FAKE"
).sum()

real_count = (
    train_df["label"] == "REAL"
).sum()

if fake_count < N_FAKE:

    raise ValueError(
        f"[ERROR] Not enough FAKE examples in TRAIN. "
        f"Required: {N_FAKE}, available: {fake_count}"
    )

if real_count < N_REAL:

    raise ValueError(
        f"[ERROR] Not enough REAL examples in TRAIN. "
        f"Required: {N_REAL}, available: {real_count}"
    )


# ============================================================
# 13. SELECT FEW-SHOT EXAMPLES FROM TRAIN ONLY
# ============================================================

fake_examples = (
    train_df[
        train_df["label"] == "FAKE"
    ]
    .sample(
        n=N_FAKE,
        random_state=RANDOM_STATE
    )
)

real_examples = (
    train_df[
        train_df["label"] == "REAL"
    ]
    .sample(
        n=N_REAL,
        random_state=RANDOM_STATE
    )
)


# Combine examples

fewshot_examples = pd.concat(
    [
        fake_examples,
        real_examples
    ],
    ignore_index=True
)


# Shuffle examples

fewshot_examples = fewshot_examples.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# 14. DISPLAY FEW-SHOT EXAMPLES
# ============================================================

print()
print("=" * 70)
print("FEW-SHOT EXAMPLES")
print("=" * 70)

print(
    f"[INFO] FAKE examples: {N_FAKE}"
)

print(
    f"[INFO] REAL examples: {N_REAL}"
)

print(
    f"[INFO] Total examples: {len(fewshot_examples)}"
)

print(
    "[INFO] Source: TRAIN dataset only"
)

for i, row in fewshot_examples.iterrows():

    print()
    print(f"Example {i + 1}")
    print("-" * 70)

    print(
        f"Statement:\n{row['statement']}"
    )

    print(
        f"Label: {row['label']}"
    )


# ============================================================
# 15. SAFETY CHECK - NO TEST OVERLAP
# ============================================================

train_example_statements = set(
    fewshot_examples["statement"]
    .astype(str)
)

test_statements = set(
    test_df["statement"]
    .astype(str)
)

overlap = (
    train_example_statements
    .intersection(test_statements)
)

if overlap:

    raise ValueError(
        f"[ERROR] {len(overlap)} few-shot examples "
        "also appear in TEST set."
    )

print()
print(
    "[OK] No few-shot examples overlap with TEST statements."
)


# ============================================================
# 16. SAVE FEW-SHOT EXAMPLES
# ============================================================

fewshot_examples[
    ["statement", "label"]
].to_csv(
    EXAMPLES_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "[OK] Few-shot examples saved to:"
)

print(
    EXAMPLES_OUTPUT_PATH
)


# ============================================================
# 17. BUILD FEW-SHOT PROMPT
# ============================================================

examples_text = ""

for i, row in fewshot_examples.iterrows():

    examples_text += f"""
Example {i + 1}:

Statement:
{row["statement"]}

Label:
{row["label"]}
""".strip()

    examples_text += "\n\n"


# ============================================================
# 18. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a fact-checking classification model.

Your task is to classify a statement as either FAKE or REAL.

FAKE:
The statement is false, misleading, substantially incorrect,
or contains a significant factual error.

REAL:
The statement is substantially true or factually supported.

You must choose exactly one label.

IMPORTANT:
Return ONLY one of these two words:

FAKE
REAL

Do not provide explanations.
Do not provide punctuation.
Do not provide any other text.
""".strip()


# ============================================================
# 19. BUILD USER PROMPT
# ============================================================

def build_prompt(statement):

    return f"""
Here are examples of previously classified statements.

FEW-SHOT EXAMPLES:

{examples_text}

Now classify the following new statement.

Statement:
{statement}

Return exactly one word:

FAKE

or

REAL
""".strip()


# ============================================================
# 20. RESPONSE PARSER
# ============================================================

def parse_prediction(response):

    if not response:

        return "INVALID"

    text = (
        response
        .strip()
        .upper()
    )

    # Exact match

    if text == "FAKE":

        return "FAKE"

    if text == "REAL":

        return "REAL"

    # Search standalone labels

    matches = re.findall(
        r"\b(FAKE|REAL)\b",
        text
    )

    if len(matches) == 1:

        return matches[0]

    return "INVALID"


# ============================================================
# 21. TEST CONFIGURATION
# ============================================================

total = len(test_df)

predictions = []
raw_responses = []

successful_requests = 0
failed_requests = 0


print()
print("=" * 70)
print("LLAMA 3.2 3B - FEW-SHOT FULL TEST")
print("=" * 70)

print(
    f"Model:                 {MODEL}"
)

print(
    "Provider:              OpenRouter"
)

print(
    "Method:                FEW-SHOT"
)

print(
    f"Few-shot examples:     {len(fewshot_examples)}"
)

print(
    f"FAKE examples:         {N_FAKE}"
)

print(
    f"REAL examples:         {N_REAL}"
)

print(
    f"Evaluation samples:    {total}"
)

print(
    "Examples source:       TRAIN"
)

print(
    "Evaluation source:     TEST"
)

print(
    f"Random state:          {RANDOM_STATE}"
)

print(
    "Temperature:           0"
)

print("=" * 70)


# ============================================================
# 22. RUN MODEL
# ============================================================

for position, (_, row) in enumerate(
    test_df.iterrows(),
    start=1
):

    statement = str(
        row["statement"]
    )

    true_label = (
        str(row["label"])
        .strip()
        .upper()
    )

    print()
    print(
        f"Sample {position}/{total}"
    )

    print("-" * 70)

    print(
        f"Statement:\n{statement}"
    )

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": build_prompt(
                        statement
                    )
                }
            ],
            temperature=0,
            max_tokens=10
        )

        raw_response = (
            response
            .choices[0]
            .message
            .content
            or ""
        )

        prediction = parse_prediction(
            raw_response
        )

        successful_requests += 1

        print(
            f"Llama raw response: {raw_response}"
        )

        print(
            f"Llama prediction:   {prediction}"
        )

    except Exception as e:

        failed_requests += 1

        raw_response = (
            f"ERROR: {str(e)}"
        )

        prediction = "INVALID"

        print(
            f"[ERROR] Request failed: {e}"
        )

    # Determine result

    if prediction == true_label:

        result = "CORRECT"

    elif prediction == "INVALID":

        result = "INVALID"

    else:

        result = "INCORRECT"

    print(
        f"True label:         {true_label}"
    )

    print(
        f"Result:             {result}"
    )

    predictions.append(
        prediction
    )

    raw_responses.append(
        raw_response
    )

    time.sleep(
        REQUEST_DELAY
    )


# ============================================================
# 23. CREATE RESULTS DATAFRAME
# ============================================================

results_df = test_df.copy()

results_df["raw_response"] = (
    raw_responses
)

results_df["prediction"] = (
    predictions
)

results_df["result"] = results_df.apply(
    lambda row:
        "CORRECT"
        if row["prediction"] == row["label"]
        else (
            "INVALID"
            if row["prediction"] == "INVALID"
            else "INCORRECT"
        ),
    axis=1
)


# ============================================================
# 24. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "[OK] Results saved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 25. VALID PREDICTIONS
# ============================================================

valid_df = results_df[
    results_df["prediction"].isin(
        ["FAKE", "REAL"]
    )
].copy()


invalid_predictions = (
    len(results_df) -
    len(valid_df)
)


correct_predictions = (
    valid_df["prediction"] ==
    valid_df["label"]
).sum()


incorrect_predictions = (
    len(valid_df) -
    correct_predictions
)


# ============================================================
# 26. PERFORMANCE METRICS
# ============================================================

if len(valid_df) > 0:

    y_true = valid_df["label"]

    y_pred = valid_df["prediction"]


    # Accuracy

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    # FAKE metrics

    precision_fake = precision_score(
        y_true,
        y_pred,
        pos_label="FAKE",
        zero_division=0
    )

    recall_fake = recall_score(
        y_true,
        y_pred,
        pos_label="FAKE",
        zero_division=0
    )

    f1_fake = f1_score(
        y_true,
        y_pred,
        pos_label="FAKE",
        zero_division=0
    )


    # REAL metrics

    precision_real = precision_score(
        y_true,
        y_pred,
        pos_label="REAL",
        zero_division=0
    )

    recall_real = recall_score(
        y_true,
        y_pred,
        pos_label="REAL",
        zero_division=0
    )

    f1_real = f1_score(
        y_true,
        y_pred,
        pos_label="REAL",
        zero_division=0
    )


    # Macro F1

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    # Confusion matrix

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "FAKE",
            "REAL"
        ]
    )


    # ========================================================
    # 27. FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    print(
        f"Total samples:             {len(results_df)}"
    )

    print(
        f"Successful requests:       {successful_requests}"
    )

    print(
        f"Failed requests:           {failed_requests}"
    )

    print(
        f"Valid predictions:         {len(valid_df)}"
    )

    print(
        f"Invalid predictions:       {invalid_predictions}"
    )

    print(
        f"Correct predictions:       {correct_predictions}"
    )

    print(
        f"Incorrect predictions:     {incorrect_predictions}"
    )


    # ========================================================
    # 28. PERFORMANCE METRICS
    # ========================================================

    print()
    print("=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    print(
        f"Accuracy:              {accuracy * 100:.2f}%"
    )

    print(
        f"Precision (FAKE):     {precision_fake * 100:.2f}%"
    )

    print(
        f"Recall (FAKE):        {recall_fake * 100:.2f}%"
    )

    print(
        f"F1-score (FAKE):      {f1_fake * 100:.2f}%"
    )

    print(
        f"Precision (REAL):     {precision_real * 100:.2f}%"
    )

    print(
        f"Recall (REAL):        {recall_real * 100:.2f}%"
    )

    print(
        f"F1-score (REAL):      {f1_real * 100:.2f}%"
    )

    print(
        f"Macro F1:             {macro_f1 * 100:.2f}%"
    )


    # ========================================================
    # 29. CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        pd.DataFrame(
            cm,
            index=[
                "Actual FAKE",
                "Actual REAL"
            ],
            columns=[
                "Predicted FAKE",
                "Predicted REAL"
            ]
        )
    )


    # ========================================================
    # 30. CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[
                "FAKE",
                "REAL"
            ],
            zero_division=0
        )
    )


# ============================================================
# 31. PREDICTION DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    results_df["prediction"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# 32. TRUE LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("TRUE LABEL DISTRIBUTION")
print("=" * 70)

print(
    results_df["label"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# 33. FINAL EXPERIMENT INFORMATION
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT INFORMATION")
print("=" * 70)

print(
    f"Model:                 {MODEL}"
)

print(
    "Provider:              OpenRouter"
)

print(
    "Dataset:               LIAR"
)

print(
    "Task:                  Binary FAKE/REAL classification"
)

print(
    "Method:                Few-Shot"
)

print(
    f"Few-shot examples:     {len(fewshot_examples)}"
)

print(
    f"FAKE examples:         {N_FAKE}"
)

print(
    f"REAL examples:         {N_REAL}"
)

print(
    f"Evaluation samples:    {len(test_df)}"
)

print(
    "Examples source:       TRAIN"
)

print(
    "Evaluation source:     TEST"
)

print(
    f"Random state:          {RANDOM_STATE}"
)

print(
    "Temperature:           0"
)

print()
print(
    "[OK] Few-shot examples were taken only from TRAIN."
)

print(
    "[OK] TEST set was reserved exclusively for evaluation."
)

print(
    "[OK] No overlap between few-shot examples and TEST set."
)


# ============================================================
# 34. COMPLETED
# ============================================================

print()
print("=" * 70)
print("FEW-SHOT FULL TEST COMPLETED")
print("=" * 70)