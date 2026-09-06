import os
import re
import time
import pandas as pd

from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from dotenv import load_dotenv


# ============================================================
# LLAMA 3.2 3B - OPENROUTER ZERO-SHOT FULL TEST
# ============================================================

print("=" * 70)
print("LLAMA 3.2 3B - OPENROUTER ZERO-SHOT FULL TEST")
print("=" * 70)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL = "meta-llama/llama-3.2-3b-instruct"

TEST_PATH = "data/processed/binary/test.csv"

OUTPUT_PATH = (
    "results/llama/zero-shot/"
    "llama_openrouter_zeroshot_full_test.csv"
)

# Delay between API requests
REQUEST_DELAY = 0.2

# Number of retries if an API request fails
MAX_RETRIES = 3

# Maximum number of output tokens
MAX_TOKENS = 10

# Deterministic generation
TEMPERATURE = 0


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("ENVIRONMENT CONFIGURATION")
print("=" * 70)

load_dotenv()

api_key = os.getenv("OPEN_ROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "[ERROR] OPEN_ROUTER_API_KEY not found in environment.\n"
        "Make sure your .env file contains:\n"
        "OPEN_ROUTER_API_KEY=your_key"
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
# 4. LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

try:

    df = pd.read_csv(TEST_PATH)

except FileNotFoundError:

    raise FileNotFoundError(
        f"[ERROR] Test dataset not found:\n{TEST_PATH}"
    )

print("[OK] Dataset loaded")
print(f"[INFO] Total samples: {len(df)}")
print(f"[INFO] Columns: {list(df.columns)}")


# ============================================================
# 5. VALIDATE DATASET
# ============================================================

print("\n" + "=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

required_columns = {"statement", "label"}

if not required_columns.issubset(df.columns):

    missing_columns = required_columns - set(df.columns)

    raise ValueError(
        f"[ERROR] Missing required columns: {missing_columns}"
    )

print("[OK] Required columns found")


# ============================================================
# 6. REMOVE MISSING VALUES
# ============================================================

initial_count = len(df)

df = df.dropna(
    subset=["statement", "label"]
).copy()

removed_count = initial_count - len(df)

if removed_count > 0:

    print(
        f"[INFO] Removed {removed_count} samples "
        "with missing statement/label"
    )

else:

    print("[OK] No missing statement/label values")


# ============================================================
# 7. NORMALIZE LABELS
# ============================================================

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# 8. CHECK BINARY LABELS
# ============================================================

valid_labels = {"FAKE", "REAL"}

found_labels = set(
    df["label"].unique()
)

invalid_labels = found_labels - valid_labels

if invalid_labels:

    raise ValueError(
        "[ERROR] Invalid labels found: "
        f"{invalid_labels}"
    )

print("[OK] Binary FAKE/REAL labels confirmed")


# ============================================================
# 9. DISPLAY DATASET DISTRIBUTION
# ============================================================

print("\n[INFO] Test dataset label distribution:")
print(
    df["label"]
    .value_counts()
)


# ============================================================
# 10. ZERO-SHOT SYSTEM PROMPT
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
- Return ONLY one of these two words:
  FAKE
  REAL
- Do not provide explanations.
- Do not provide reasoning.
- Do not provide punctuation.
- Do not provide any other text.
""".strip()


# ============================================================
# 11. USER PROMPT
# ============================================================

def build_prompt(statement):

    return f"""
Classify the following statement.

Statement:
{statement}

Answer with exactly one word:
FAKE
or
REAL
""".strip()


# ============================================================
# 12. RESPONSE PARSER
# ============================================================

def parse_prediction(response):

    if not response:

        return "INVALID"

    text = response.strip().upper()

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if text == "FAKE":

        return "FAKE"

    if text == "REAL":

        return "REAL"

    # --------------------------------------------------------
    # Search for standalone labels
    # --------------------------------------------------------

    matches = re.findall(
        r"\b(FAKE|REAL)\b",
        text
    )

    # Exactly one valid label found
    if len(matches) == 1:

        return matches[0]

    # More than one label or no valid label
    return "INVALID"


# ============================================================
# 13. CLASSIFICATION FUNCTION WITH RETRIES
# ============================================================

def classify_statement(statement):

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            completion = client.chat.completions.create(
                model=MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": build_prompt(statement)
                    }
                ],

                temperature=TEMPERATURE,

                max_tokens=MAX_TOKENS
            )

            raw_response = (
                completion
                .choices[0]
                .message
                .content
                or ""
            )

            prediction = parse_prediction(
                raw_response
            )

            return raw_response, prediction, None

        except Exception as e:

            last_error = e

            print(
                f"[WARNING] Request failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            print(
                f"[WARNING] {type(e).__name__}: {e}"
            )

            if attempt < MAX_RETRIES:

                # Increasing delay after failure
                retry_delay = attempt * 2

                print(
                    f"[INFO] Retrying in "
                    f"{retry_delay} seconds..."
                )

                time.sleep(retry_delay)

    return (
        f"ERROR: {str(last_error)}",
        "INVALID",
        last_error
    )


# ============================================================
# 14. TEST CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT CONFIGURATION")
print("=" * 70)

print(f"Model:             {MODEL}")
print("Provider:          OpenRouter")
print("Dataset:           LIAR")
print("Task:              Binary classification")
print("Method:            ZERO-SHOT")
print(f"Test samples:      {len(df)}")
print(f"Temperature:       {TEMPERATURE}")
print(f"Max tokens:        {MAX_TOKENS}")
print(f"Request delay:     {REQUEST_DELAY}s")
print(f"Max retries:       {MAX_RETRIES}")

print("\n[IMPORTANT]")
print("- No training examples are provided to the model.")
print("- No few-shot examples are used.")
print("- The test set is used only for evaluation.")


# ============================================================
# 15. RUN FULL TEST
# ============================================================

print("\n" + "=" * 70)
print("STARTING ZERO-SHOT FULL TEST")
print("=" * 70)


predictions = []
raw_responses = []
results = []

successful_requests = 0
failed_requests = 0

total = len(df)


for position, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    statement = str(
        row["statement"]
    )

    true_label = str(
        row["label"]
    ).strip().upper()


    print("\n" + "-" * 70)
    print(
        f"Sample {position}/{total}"
    )
    print("-" * 70)

    print(
        f"Statement:\n{statement}"
    )

    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    raw_response, prediction, error = (
        classify_statement(statement)
    )

    # --------------------------------------------------------
    # REQUEST STATUS
    # --------------------------------------------------------

    if error is None:

        successful_requests += 1

    else:

        failed_requests += 1


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if prediction == "INVALID":

        result = "INVALID"

    elif prediction == true_label:

        result = "CORRECT"

    else:

        result = "INCORRECT"


    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        f"\nLlama raw response: "
        f"{raw_response}"
    )

    print(
        f"Llama prediction:   "
        f"{prediction}"
    )

    print(
        f"True label:         "
        f"{true_label}"
    )

    print(
        f"Result:             "
        f"{result}"
    )


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    predictions.append(
        prediction
    )

    raw_responses.append(
        raw_response
    )

    results.append(
        result
    )


    # --------------------------------------------------------
    # DELAY
    # --------------------------------------------------------

    if position < total:

        time.sleep(
            REQUEST_DELAY
        )


# ============================================================
# 16. CREATE RESULTS DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("CREATING RESULTS DATAFRAME")
print("=" * 70)

results_df = df.copy()

results_df["raw_response"] = (
    raw_responses
)

results_df["prediction"] = (
    predictions
)

results_df["result"] = (
    results
)

print("[OK] Results dataframe created")


# ============================================================
# 17. SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

output_directory = os.path.dirname(
    OUTPUT_PATH
)

os.makedirs(
    output_directory,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("[OK] Results saved to:")
print(OUTPUT_PATH)


# ============================================================
# 18. BASIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST SUMMARY")
print("=" * 70)

total_samples = len(
    results_df
)

valid_predictions_df = results_df[
    results_df["prediction"].isin(
        ["FAKE", "REAL"]
    )
].copy()

valid_predictions = len(
    valid_predictions_df
)

invalid_predictions = (
    total_samples
    - valid_predictions
)

correct_predictions = (
    results_df["result"] == "CORRECT"
).sum()

incorrect_predictions = (
    results_df["result"] == "INCORRECT"
).sum()


print(
    f"Total samples:             "
    f"{total_samples}"
)

print(
    f"Successful requests:       "
    f"{successful_requests}"
)

print(
    f"Failed requests:           "
    f"{failed_requests}"
)

print(
    f"Valid predictions:         "
    f"{valid_predictions}"
)

print(
    f"Invalid predictions:       "
    f"{invalid_predictions}"
)

print(
    f"Correct predictions:       "
    f"{correct_predictions}"
)

print(
    f"Incorrect predictions:     "
    f"{incorrect_predictions}"
)


# ============================================================
# 19. PERFORMANCE METRICS
# ============================================================

if valid_predictions > 0:

    y_true = (
        valid_predictions_df["label"]
    )

    y_pred = (
        valid_predictions_df["prediction"]
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    # --------------------------------------------------------
    # FAKE metrics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # REAL metrics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    print(
        f"Accuracy:              "
        f"{accuracy * 100:.2f}%"
    )

    print()

    print(
        f"Precision (FAKE):      "
        f"{precision_fake * 100:.2f}%"
    )

    print(
        f"Recall (FAKE):         "
        f"{recall_fake * 100:.2f}%"
    )

    print(
        f"F1-score (FAKE):       "
        f"{f1_fake * 100:.2f}%"
    )

    print()

    print(
        f"Precision (REAL):      "
        f"{precision_real * 100:.2f}%"
    )

    print(
        f"Recall (REAL):         "
        f"{recall_real * 100:.2f}%"
    )

    print(
        f"F1-score (REAL):       "
        f"{f1_real * 100:.2f}%"
    )

    print()

    print(
        f"Macro F1:              "
        f"{macro_f1 * 100:.2f}%"
    )


    # ========================================================
    # 20. CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["FAKE", "REAL"]
    )


    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm_df = pd.DataFrame(
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

    print()
    print(cm_df)


    # ========================================================
    # 21. CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print()

    print(
        classification_report(
            y_true,
            y_pred,
            labels=["FAKE", "REAL"],
            zero_division=0
        )
    )


else:

    print(
        "\n[WARNING] No valid predictions available."
    )

    print(
        "[WARNING] Performance metrics cannot be calculated."
    )


# ============================================================
# 22. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

print()

print(
    results_df["prediction"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# 23. TRUE LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRUE LABEL DISTRIBUTION")
print("=" * 70)

print()

print(
    results_df["label"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# 24. RESULT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RESULT DISTRIBUTION")
print("=" * 70)

print()

print(
    results_df["result"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# 25. FINAL EXPERIMENT INFORMATION
# ============================================================

print("\n" + "=" * 70)
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
    "Method:                Zero-Shot Prompting"
)

print(
    f"Evaluation samples:    {total_samples}"
)

print(
    "Training:              None"
)

print(
    "Few-shot examples:     None"
)

print(
    f"Temperature:            {TEMPERATURE}"
)

print(
    f"Max tokens:             {MAX_TOKENS}"
)

print(
    f"Request delay:          {REQUEST_DELAY}s"
)

print(
    f"Max retries:            {MAX_RETRIES}"
)

print(
    "\n[OK] Test set was used only for evaluation."
)

print(
    "[OK] No few-shot examples were provided."
)

print(
    "[OK] Zero-shot experiment completed."
)


# ============================================================
# 26. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("LLAMA 3.2 3B ZERO-SHOT FULL TEST COMPLETED")
print("=" * 70)