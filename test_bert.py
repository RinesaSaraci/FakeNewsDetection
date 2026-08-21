from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "bert-base-uncased"

print("Loading tokenizer...")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

print("Loading BERT model...")

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("BERT loaded successfully!")