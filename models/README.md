## Trained Models

This project uses both classical ML models and a fine-tuned DistilBERT model.

🚫 Trained model files are NOT included in this repository due to size limits.

---

## How to Generate Models

### Classical Models
Run:
- `notebooks/01logisticregression.ipynb`

This will generate:
- Logistic Regression
- Linear SVM
- TF-IDF Vectorizer

### DistilBERT
Run:
- `notebooks/02distilberttraining.ipynb`

The trained model will be saved to:
models/distilbert-combined-v2/


---

## Model Summary
- Task: Fake News Classification
- Approaches:
  - TF-IDF + ML
  - DistilBERT (fine-tuned)
