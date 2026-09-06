import os
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
# LLAMA 3.2 3B - OPENROUTER ZERO-SHOT 100 SAMPLE TEST
# ============================================================

print("=" * 70)
print("LLAMA 3.2 3B - OPENROUTER ZERO-SHOT TEST (100 SAMPLES)")
print("=" * 70)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPEN_ROUTER_API_KEY")

if not api_key:
    print("\nERROR: OPENROUTER_API_KEY was not found!")
    print("Make sure your .env file contains:")
    print("OPEN_ROUTER_API_KEY=sk-or-v1-xxxxxxxx")
    exit(1)

print("\n[OK] OPEN_ROUTER_API_KEY found")


# ============================================================
# 2. CREATE OPENROUTER CLIENT
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

print("[OK] OpenRouter client created")


# ============================================================
# 3. LOAD DATASET
# ============================================================

DATASET_PATH = "data/processed/binary/test.csv"

try:
    df = pd.read_csv(DATASET_PATH)

except FileNotFoundError:
    print("\nERROR: Dataset not found:")
    print(DATASET_PATH)
    print("\nChange DATASET_PATH in the script.")
    exit(1)


print("[OK] Dataset loaded")
print(f"[INFO] Total samples: {len(df)}")
print(f"[INFO] Columns: {list(df.columns)}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["statement", "label"]

for column in required_columns:

    if column not in df.columns:

        print(f"\nERROR: Required column '{column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        exit(1)

print("[OK] Required columns found")


# ============================================================
# 5. NORMALIZE LABELS
# ============================================================

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.upper()
)


print("\n[INFO] Dataset label distribution:")
print(df["label"].value_counts())


# ============================================================
# 6. CHECK BINARY LABELS
# ============================================================

expected_labels = {"FAKE", "REAL"}

actual_labels = set(df["label"].unique())

if not actual_labels.issubset(expected_labels):

    print("\nERROR: Dataset contains unexpected labels!")
    print(f"Expected: {expected_labels}")
    print(f"Found: {actual_labels}")
    exit(1)

print("[OK] Binary FAKE/REAL labels confirmed")


# ============================================================
# 7. SELECT 100 RANDOM TEST SAMPLES
# ============================================================

TEST_SAMPLES = 100
RANDOM_STATE = 42

if len(df) < TEST_SAMPLES:

    print(
        f"\nERROR: Dataset contains only {len(df)} samples."
    )
    print(
        f"Cannot select {TEST_SAMPLES} samples."
    )
    exit(1)


test_df = df.sample(
    n=TEST_SAMPLES,
    random_state=RANDOM_STATE
).copy()


print(f"\n[INFO] Testing {len(test_df)} samples")
print("[INFO] Method: ZERO-SHOT")
print("[INFO] Examples provided to Llama: 0")
print(f"[INFO] Random state: {RANDOM_STATE}")


print("\n[INFO] Selected test-set label distribution:")
print(test_df["label"].value_counts())


# ============================================================
# 8. ZERO-SHOT PROMPT
# ============================================================

def classify_statement(statement):

    prompt = f"""
You are performing binary classification of claims from a
fact-checking dataset.

Your task is to classify the claim as either FAKE or REAL.

FAKE:
The claim is false, misleading, or contains a false factual assertion.

REAL:
The claim is substantially true or supported by factual evidence.

Important instructions:
- Classify the factual claim itself.
- Do not classify based on whether you agree or disagree with the
  person, politician, party, organization, or opinion mentioned.
- Do not use the political identity of the speaker as a reason for
  classification.
- Do not provide an explanation.
- Do not provide probabilities.
- Do not provide additional text.

Statement:
{statement}

Respond with exactly one word:

FAKE

or

REAL
"""

    completion = client.chat.completions.create(

        model="meta-llama/llama-3.2-3b-instruct",

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

    response_upper = response.upper()

    # Exact valid response
    if response_upper == "FAKE":
        return "FAKE"

    if response_upper == "REAL":
        return "REAL"

    # Handle cases such as:
    # "FAKE."
    # "REAL\n"
    cleaned = response_upper.strip(" .,!?:;")

    if cleaned == "FAKE":
        return "FAKE"

    if cleaned == "REAL":
        return "REAL"

    return "INVALID"


# ============================================================
# 9. RUN TEST
# ============================================================

results = []

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)


for counter, (index, row) in enumerate(
    test_df.iterrows(),
    start=1
):

    statement = str(row["statement"])
    true_label = str(row["label"]).strip().upper()

    try:

        prediction = classify_statement(statement)

        if prediction == "INVALID":

            result = "INVALID"

        elif prediction == true_label:

            result = "CORRECT"

        else:

            result = "INCORRECT"


        print(f"\nSample {counter}/{len(test_df)}")
        print("-" * 70)

        print("Statement:")
        print(statement)

        print(f"\nTrue label:        {true_label}")
        print(f"Llama prediction:  {prediction}")
        print(f"Result:             {result}")


        results.append({

            "index": index,

            "statement": statement,

            "true_label": true_label,

            "prediction": prediction,

            "result": result

        })


    except Exception as e:

        print(f"\nERROR processing sample {counter}")

        print(f"Error type: {type(e).__name__}")

        print(f"Error: {e}")


        results.append({

            "index": index,

            "statement": statement,

            "true_label": true_label,

            "prediction": "ERROR",

            "result": "ERROR"

        })


# ============================================================
# 10. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 11. BASIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ZERO-SHOT TEST SUMMARY")
print("=" * 70)


total = len(results_df)

successful = (
    results_df["prediction"] != "ERROR"
).sum()

failed = total - successful

valid_predictions = results_df["prediction"].isin(
    ["FAKE", "REAL"]
)

valid_count = valid_predictions.sum()

invalid_count = (~valid_predictions).sum()

correct = (
    results_df["result"] == "CORRECT"
).sum()

incorrect = (
    results_df["result"] == "INCORRECT"
).sum()


print(f"Total samples:             {total}")
print(f"Successful requests:       {successful}")
print(f"Failed requests:           {failed}")

print(f"Valid predictions:         {valid_count}")
print(f"Invalid predictions:       {invalid_count}")

print(f"Correct predictions:       {correct}")
print(f"Incorrect predictions:     {incorrect}")


# ============================================================
# 12. CALCULATE METRICS
# ============================================================

valid_results = results_df[
    results_df["prediction"].isin(["FAKE", "REAL"])
].copy()


if len(valid_results) > 0:

    y_true = valid_results["true_label"]
    y_pred = valid_results["prediction"]


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        pos_label="FAKE",
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        pos_label="FAKE",
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
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
    print(f"Precision (FAKE):     {precision * 100:.2f}%")
    print(f"Recall (FAKE):        {recall * 100:.2f}%")
    print(f"F1-score (FAKE):      {f1 * 100:.2f}%")
    print(f"Macro F1:             {macro_f1 * 100:.2f}%")


    # ========================================================
    # 13. CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["FAKE", "REAL"]
    )


    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print("\n                  Predicted")
    print("                 FAKE    REAL")
    print(
        f"Actual FAKE      {cm[0][0]:<7} {cm[0][1]}"
    )
    print(
        f"Actual REAL      {cm[1][0]:<7} {cm[1][1]}"
    )


    # ========================================================
    # 14. CLASSIFICATION REPORT
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
# 15. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    results_df["prediction"].value_counts(
        dropna=False
    )
)


# ============================================================
# 16. TRUE LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRUE LABEL DISTRIBUTION")
print("=" * 70)

print(
    results_df["true_label"].value_counts(
        dropna=False
    )
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

OUTPUT_PATH = "llama_openrouter_zeroshot_100_test.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("\n[OK] Results saved to:")
print(OUTPUT_PATH)


# ============================================================
# 18. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("OPENROUTER ZERO-SHOT 100-SAMPLE TEST COMPLETED")
print("=" * 70)