import os
import joblib
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ------------------------------------------------------------------
# Base directory (project root)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# Load classical ML models
# ------------------------------------------------------------------
tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
svm = joblib.load(os.path.join(BASE_DIR, "models", "linear_svm.pkl"))
lr = joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression.pkl"))

# ------------------------------------------------------------------
# Load DistilBERT v2 (headline + article trained)
# ------------------------------------------------------------------
device = torch.device("cpu")  # keep CPU for now

BERT_PATH = os.path.join(BASE_DIR, "models", "distilbert-combined-v2")

tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PATH)

bert = DistilBertForSequenceClassification.from_pretrained(
    BERT_PATH
).to(device)

bert.eval()

# ------------------------------------------------------------------
# Prediction functions
# ------------------------------------------------------------------
def predict_classical(text: str):
    vec = tfidf.transform([text])
    return {
        "svm": int(svm.predict(vec)[0]),
        "lr": int(lr.predict(vec)[0])
    }

def predict_bert(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = bert(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    probs = probs.squeeze().tolist()

    # Debug (keep for now)
    print("BERT RAW PROBS:", probs)
    print("BERT ID2LABEL:", bert.config.id2label)

    return probs
