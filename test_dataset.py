import pandas as pd

# Paths
train_path = "data/raw/liar_dataset/train.tsv"
valid_path = "data/raw/liar_dataset/valid.tsv"
test_path = "data/raw/liar_dataset/test.tsv"

# Read datasets
train_df = pd.read_csv(train_path, sep="\t", header=None)
valid_df = pd.read_csv(valid_path, sep="\t", header=None)
test_df = pd.read_csv(test_path, sep="\t", header=None)

# Basic information
print("===== DATASET SIZES =====")
print("Train:", train_df.shape)
print("Validation:", valid_df.shape)
print("Test:", test_df.shape)

print("\n===== NUMBER OF COLUMNS =====")
print("Train columns:", len(train_df.columns))
print("Validation columns:", len(valid_df.columns))
print("Test columns:", len(test_df.columns))

print("\n===== FIRST 3 TRAINING EXAMPLES =====")
print(train_df.head(3).to_string())

print("\n===== LABEL DISTRIBUTION =====")
print("Train:")
print(train_df[1].value_counts())

print("\nValidation:")
print(valid_df[1].value_counts())

print("\nTest:")
print(test_df[1].value_counts())

print("\n===== MISSING VALUES =====")
print("Train:")
print(train_df.isnull().sum())

print("\nValidation:")
print(valid_df.isnull().sum())

print("\nTest:")
print(test_df.isnull().sum())