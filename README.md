# Fake News Classification System

An end-to-end fake news classification backend built using classical NLP models and transformer-based deep learning, with a confidence-aware ensemble to mitigate dataset bias and overconfidence.

---

## 🚀 Overview

This project classifies news articles as **REAL** or **FAKE** based on linguistic and stylistic patterns rather than factual verification.

It combines:
- Classical ML models for stability
- A transformer model for contextual understanding
- A gated ensemble strategy for robust decision-making

The system is exposed via a **FastAPI** backend with a clean, modular architecture.

---

## 🧠 Model Architecture

### Classical Models (Baseline)
- TF-IDF Vectorizer
- Logistic Regression
- Linear SVM

These models provide stable and interpretable predictions and act as the fallback decision-makers.

### Transformer Model
- DistilBERT (fine-tuned on news articles with headline + body text)
- Used only when confident to avoid overconfident misclassification

### Ensemble Strategy
- If BERT confidence ≥ threshold → trust BERT
- Otherwise → fallback to majority vote from classical models

This confidence-aware gating prevents coin-flip decisions and reduces bias from training data.

---

## 🏗️ Project Structure

backend/
├── main.py # FastAPI app
├── inference.py # Model loading & prediction
├── ensemble.py # Final decision logic

models/
├── tfidf_vectorizer.pkl
├── linear_svm.pkl
├── logistic_regression.pkl
├── distilbert-combined-v2/


---

## ⚙️ Running Locally

### Install dependencies
```bash
pip install -r requirements.txt

##Start the server

uvicorn backend.main:app --reload

##Open Swagger UI

http://127.0.0.1:8000/docs



⚠️ Limitations

Does not verify factual correctness

Predictions are based on linguistic patterns

Performance depends on similarity to training data

Headline-only inputs may be uncertain


Future Improvements

Separate headline vs article models

Knowledge-based fact verification

Advanced calibration techniques


