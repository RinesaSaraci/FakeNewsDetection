import os
import re
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# LLAMA 3.2 3B - FEW-SHOT ZERO-SHOT COMPARISON
# ============================================================

print("=" * 70)
print("LLAMA 3.2 3B - OPENROUTER FEW-SHOT TEST")
print("=" * 70)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")

if not openrouter_api_key:
    print("\nERROR: OPEN_ROUTER_API_KEY was not found!")
    print("Make sure your .env file contains:")
    print("OPEN_ROUTER_API_KEY=sk-or-xxxxxxxx")
    exit(1)

print("[OK] OPEN_ROUTER_API_KEY found")


# ============================================================
# 2. CREATE OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

print("[OK] OpenRouter client created")


# ============================================================
# 3. CONFIGURATION
# ============================================================

TRAIN_DATASET_PATH = "data/processed/binary/train.csv"
TEST_DATASET_PATH = "data/processed/binary/test.csv"

OUTPUT_DIR = "results/few-shot"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "llama_openrouter_fewshot_100_test.csv"
)

MODEL_NAME = "meta-llama/llama-3.2-3b-instruct"

TEST_SAMPLES = 100

# Total few-shot examples
FEW_SHOT_PER_CLASS = 3

RANDOM_STATE = 42


# ============================================================
# 4. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[OK] Output directory ready: {OUTPUT_DIR}")


# ============================================================
# 5. LOAD TRAIN DATASET
# ============================================================

try:
    train_df = pd.read_csv(TRAIN_DATASET_PATH)

except FileNotFoundError:
    print("\nERROR: Training dataset not found:")
    print(TRAIN_DATASET_PATH)
    exit(1)

print("\n[OK] Training dataset loaded")
print(f"[INFO] Training samples: {len(train_df)}")
print(f"[INFO] Train columns: {list(train_df.columns)}")


# ============================================================
# 6. LOAD TEST DATASET
# ============================================================

try:
    test_df = pd.read_csv(TEST_DATASET_PATH)

except FileNotFoundError:
    print("\nERROR: Test dataset not found:")
    print(TEST_DATASET_PATH)
    exit(1)

print("\n[OK] Test dataset loaded")
print(f"[INFO] Test samples: {len(test_df)}")
print(f"[INFO] Test columns: {list(test_df.columns)}")


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["statement", "label"]

for column in required_columns:

    if column not in train_df.columns:
        print(f"\nERROR: Required column '{column}' not found in TRAIN!")
        print(f"Available columns: {list(train_df.columns)}")
        exit(1)

    if column not in test_df.columns:
        print(f"\nERROR: Required column '{column}' not found in TEST!")
        print(f"Available columns: {list(test_df.columns)}")
        exit(1)

print("\n[OK] Required columns found in both datasets")


# ============================================================
# 8. NORMALIZE LABELS
# ============================================================

train_df["label"] = (
    train_df["label"]
    .astype(str)
    .str.strip()
    .str.upper()
)

test_df["label"] = (
    test_df["label"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# 9. CHECK BINARY LABELS
# ============================================================

valid_labels = {"FAKE", "REAL"}

train_labels = set(train_df["label"].unique())
test_labels = set(test_df["label"].unique())

if not train_labels.issubset(valid_labels):
    print("\nERROR: Training dataset contains unexpected labels:")
    print(train_labels)
    exit(1)

if not test_labels.issubset(valid_labels):
    print("\nERROR: Test dataset contains unexpected labels:")
    print(test_labels)
    exit(1)

print("\n[OK] Binary FAKE/REAL labels confirmed")


# ============================================================
# 10. DATASET DISTRIBUTION
# ============================================================

print("\n[INFO] TRAIN label distribution:")
print(train_df["label"].value_counts())

print("\n[INFO] TEST label distribution:")
print(test_df["label"].value_counts())


# ============================================================
# 11. SELECT FEW-SHOT EXAMPLES FROM TRAIN ONLY
# ============================================================

fake_examples = (
    train_df[train_df["label"] == "FAKE"]
    .sample(
        n=FEW_SHOT_PER_CLASS,
        random_state=RANDOM_STATE
    )
)

real_examples = (
    train_df[train_df["label"] == "REAL"]
    .sample(
        n=FEW_SHOT_PER_CLASS,
        random_state=RANDOM_STATE
    )
)


# Combine examples
few_shot_df = pd.concat(
    [fake_examples, real_examples],
    ignore_index=True
)

# Shuffle examples so they are not always FAKE first
few_shot_df = few_shot_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# 12. DISPLAY FEW-SHOT EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("FEW-SHOT EXAMPLES")
print("=" * 70)

print(
    f"[INFO] Examples per class: {FEW_SHOT_PER_CLASS}"
)

print(
    f"[INFO] Total examples: {len(few_shot_df)}"
)

print(
    "[INFO] Source: TRAINING SET ONLY"
)

print(
    "[INFO] Test set is NOT used for examples"
)

for i, row in few_shot_df.iterrows():

    print(f"\nExample {i + 1}")
    print("-" * 70)
    print(f"Statement: {row['statement']}")
    print(f"Label:     {row['label']}")


# ============================================================
# 13. SELECT TEST SAMPLES
# ============================================================

# Use exactly the same reproducible 100-sample
# selection strategy as the previous experiment.

test_df = test_df.sample(
    n=TEST_SAMPLES,
    random_state=RANDOM_STATE
).reset_index(drop=False)

print("\n" + "=" * 70)
print("TEST CONFIGURATION")
print("=" * 70)

print(f"[INFO] Testing {len(test_df)} samples")
print("[INFO] Method: FEW-SHOT")
print(f"[INFO] Few-shot examples: {len(few_shot_df)}")
print(f"[INFO] Examples per class: {FEW_SHOT_PER_CLASS}")
print("[INFO] Examples source: TRAIN")
print("[INFO] Evaluation source: TEST")
print(f"[INFO] Random state: {RANDOM_STATE}")

print("\n[INFO] Selected test-set label distribution:")
print(test_df["label"].value_counts())


# ============================================================
# 14. BUILD FEW-SHOT EXAMPLES TEXT
# ============================================================

def build_few_shot_examples():

    examples_text = ""

    for i, row in few_shot_df.iterrows():

        examples_text += f"""
Example {i + 1}:
Statement: {row['statement']}
Label: {row['label']}
"""

    return examples_text


FEW_SHOT_EXAMPLES_TEXT = build_few_shot_examples()


# ============================================================
# 15. FEW-SHOT PROMPT
# ============================================================

def classify_statement(statement):

    prompt = f"""
You are a binary fake news classification model.

Your task is to classify a new statement as either FAKE or REAL.

IMPORTANT:
- FAKE means the statement is false, misleading, inaccurate,
  unsupported, or substantially incorrect.
- REAL means the statement is substantially true or factually supported.
- Use the examples below only as demonstrations of the classification task.
- Do not explain your answer.
- Respond with exactly one word: FAKE or REAL.

FEW-SHOT EXAMPLES:
{FEW_SHOT_EXAMPLES_TEXT}

NEW STATEMENT:
{statement}

CLASSIFICATION:
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=5
    )

    response = completion.choices[0].message.content.strip()

    return response


# ============================================================
# 16. EXTRACT LABEL FROM MODEL RESPONSE
# ============================================================

def extract_label(response):

    if not response:
        return "INVALID"

    response = response.upper().strip()

    # Exact match
    if response == "FAKE":
        return "FAKE"

    if response == "REAL":
        return "REAL"

    # Search inside response
    fake_match = re.search(r"\bFAKE\b", response)
    real_match = re.search(r"\bREAL\b", response)

    if fake_match and real_match:

        # Use whichever label appears first
        if fake_match.start() < real_match.start():
            return "FAKE"
        else:
            return "REAL"

    if fake_match:
        return "FAKE"

    if real_match:
        return "REAL"

    return "INVALID"


# ============================================================
# 17. TEST EACH STATEMENT
# ============================================================

results = []

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)


for position, row in test_df.iterrows():

    statement = str(row["statement"])

    true_label = (
        str(row["label"])
        .strip()
        .upper()
    )

    original_index = row["index"]

    try:

        raw_prediction = classify_statement(statement)

        prediction = extract_label(raw_prediction)

        if prediction == "INVALID":

            result = "INVALID"

        elif prediction == true_label:

            result = "CORRECT"

        else:

            result = "INCORRECT"

        print(f"\nSample {position + 1}/{len(test_df)}")
        print("-" * 70)

        print("Statement:")
        print(statement)

        print(f"\nTrue label:        {true_label}")
        print(f"Llama raw response: {raw_prediction}")
        print(f"Llama prediction:  {prediction}")
        print(f"Result:             {result}")

        results.append({
            "index": original_index,
            "statement": statement,
            "true_label": true_label,
            "raw_response": raw_prediction,
            "prediction": prediction,
            "result": result
        })

    except Exception as e:

        print(f"\nERROR processing sample {position + 1}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")

        results.append({
            "index": original_index,
            "statement": statement,
            "true_label": true_label,
            "raw_response": "ERROR",
            "prediction": "ERROR",
            "result": "ERROR"
        })


# ============================================================
# 18. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 19. BASIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ZERO-SHOT TEST SUMMARY")
print("=" * 70)

total = len(results_df)

successful = (
    ~results_df["prediction"].isin(["ERROR"])
).sum()

failed = (
    results_df["prediction"] == "ERROR"
).sum()

valid_predictions = (
    results_df["prediction"].isin(["FAKE", "REAL"])
)

valid_count = valid_predictions.sum()

invalid_count = (~valid_predictions).sum()

correct_count = (
    results_df["result"] == "CORRECT"
).sum()

incorrect_count = (
    results_df["result"] == "INCORRECT"
).sum()


print(f"Total samples:             {total}")
print(f"Successful requests:       {successful}")
print(f"Failed requests:           {failed}")
print(f"Valid predictions:         {valid_count}")
print(f"Invalid predictions:       {invalid_count}")
print(f"Correct predictions:       {correct_count}")
print(f"Incorrect predictions:     {incorrect_count}")


# ============================================================
# 20. PERFORMANCE METRICS
# ============================================================

valid_results_df = results_df[
    results_df["prediction"].isin(["FAKE", "REAL"])
].copy()


if len(valid_results_df) > 0:

    y_true = valid_results_df["true_label"]
    y_pred = valid_results_df["prediction"]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision_fake = precision_score(
        y_true,
        y_pred,
        labels=["FAKE"],
        average="binary",
        pos_label="FAKE",
        zero_division=0
    )

    recall_fake = recall_score(
        y_true,
        y_pred,
        labels=["FAKE"],
        average="binary",
        pos_label="FAKE",
        zero_division=0
    )

    f1_fake = f1_score(
        y_true,
        y_pred,
        labels=["FAKE"],
        average="binary",
        pos_label="FAKE",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    print(f"Accuracy:              {accuracy * 100:.2f}%")
    print(f"Precision (FAKE):     {precision_fake * 100:.2f}%")
    print(f"Recall (FAKE):        {recall_fake * 100:.2f}%")
    print(f"F1-score (FAKE):      {f1_fake * 100:.2f}%")
    print(f"Macro F1:             {macro_f1 * 100:.2f}%")


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["FAKE", "REAL"]
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\n                  Predicted"
    )

    print(
        "                 FAKE    REAL"
    )

    print(
        f"Actual FAKE      {cm[0][0]:4d}    {cm[0][1]:4d}"
    )

    print(
        f"Actual REAL      {cm[1][0]:4d}    {cm[1][1]:4d}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=["FAKE", "REAL"],
            zero_division=0
        )
    )


# ============================================================
# 21. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    results_df["prediction"]
    .value_counts(dropna=False)
)


# ============================================================
# 22. TRUE LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRUE LABEL DISTRIBUTION")
print("=" * 70)

print(
    results_df["true_label"]
    .value_counts(dropna=False)
)


# ============================================================
# 23. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("\n[OK] Results saved to:")
print(OUTPUT_PATH)


# ============================================================
# 24. SAVE FEW-SHOT EXAMPLES
# ============================================================

EXAMPLES_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "llama_openrouter_fewshot_examples.csv"
)

few_shot_df[
    ["statement", "label"]
].to_csv(
    EXAMPLES_OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("[OK] Few-shot examples saved to:")
print(EXAMPLES_OUTPUT_PATH)


# ============================================================
# 25. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FEW-SHOT TEST COMPLETED")
print("=" * 70)

print("\nExperimental setup:")
print(f"- Model: {MODEL_NAME}")
print("- Provider: OpenRouter")
print("- Dataset: LIAR")
print("- Task: Binary classification")
print("- Method: Few-shot prompting")
print(f"- Few-shot examples: {len(few_shot_df)}")
print(f"- FAKE examples: {FEW_SHOT_PER_CLASS}")
print(f"- REAL examples: {FEW_SHOT_PER_CLASS}")
print(f"- Evaluation samples: {TEST_SAMPLES}")
print("- Examples source: TRAIN")
print("- Evaluation source: TEST")
print(f"- Random state: {RANDOM_STATE}")

print("\n[OK] No test samples were used as few-shot examples.")
print("[OK] Test set was reserved for evaluation.")

