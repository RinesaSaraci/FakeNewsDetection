import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LLAMA 3.2 3B - OPENROUTER ZERO-SHOT SMALL TEST
# ============================================================

print("=" * 70)
print("LLAMA 3.2 3B - OPENROUTER ZERO-SHOT TEST")
print("=" * 70)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPEN_ROUTER_API_KEY")

if not api_key:
    print("\nERROR: OPENROUTER_API_KEY was not found!")
    print("Make sure your .env file contains:")
    print("OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx")
    exit(1)

print("\n[OK] OPENROUTER_API_KEY found")


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
# 5. CHECK LABELS
# ============================================================

df["label"] = df["label"].astype(str).str.strip().str.upper()

print("\n[INFO] Dataset label distribution:")
print(df["label"].value_counts())


expected_labels = {"FAKE", "REAL"}

actual_labels = set(df["label"].unique())

if not actual_labels.issubset(expected_labels):

    print("\nERROR: Dataset contains unexpected labels!")
    print(f"Expected: {expected_labels}")
    print(f"Found: {actual_labels}")
    exit(1)

print("[OK] Binary FAKE/REAL labels confirmed")


# ============================================================
# 6. SELECT SMALL TEST SET
# ============================================================

TEST_SAMPLES = 20
RANDOM_STATE = 42

test_df = df.sample(
    n=min(TEST_SAMPLES, len(df)),
    random_state=RANDOM_STATE
).copy()

print(f"\n[INFO] Testing {len(test_df)} samples")
print("[INFO] Method: ZERO-SHOT")
print("[INFO] Examples provided to Llama: 0")
print(f"[INFO] Random state: {RANDOM_STATE}")


# ============================================================
# 7. ZERO-SHOT PROMPT
# ============================================================

def classify_statement(statement):

    prompt = f"""
You are a fake news classification model.

Your task is to classify the following statement as either FAKE or REAL.

Definitions:

FAKE:
The statement is false, misleading, or unsupported by facts.

REAL:
The statement is true or substantially supported by facts.

Statement:
{statement}

Respond with exactly one word:
FAKE
or
REAL

Do not provide an explanation.
Do not provide any other text.
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

    # Normalize response
    response_upper = response.upper()

    if response_upper == "FAKE":
        return "FAKE"

    if response_upper == "REAL":
        return "REAL"

    return "INVALID"


# ============================================================
# 8. TEST EACH STATEMENT
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

        if prediction == "ERROR":

            result = "ERROR"

        elif prediction == "INVALID":

            result = "INVALID"

        elif prediction == true_label:

            result = "CORRECT"

        else:

            result = "INCORRECT"


        print(f"\nSample {counter}/{len(test_df)}")
        print("-" * 70)

        print("Statement:")
        print(statement)

        print(f"\nTrue label:       {true_label}")
        print(f"Llama prediction: {prediction}")
        print(f"Result:            {result}")


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
# 9. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 10. BASIC TEST SUMMARY
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
# 11. CALCULATE ACCURACY
# ============================================================

if valid_count > 0:

    accuracy = correct / valid_count * 100

    print(f"Accuracy (valid only):     {accuracy:.2f}%")

else:

    print("Accuracy:                   N/A")


# ============================================================
# 12. PREDICTION DISTRIBUTION
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
# 13. TRUE LABEL DISTRIBUTION
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
# 14. SAVE RESULTS
# ============================================================

OUTPUT_PATH = "llama_openrouter_zeroshot_small_test.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("\n[OK] Results saved to:")
print(OUTPUT_PATH)


# ============================================================
# 15. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("OPENROUTER ZERO-SHOT SMALL TEST COMPLETED")
print("=" * 70)