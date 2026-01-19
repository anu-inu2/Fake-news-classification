from fastapi import FastAPI
from pydantic import BaseModel
from backend.inference import predict_classical, predict_bert
from backend.ensemble import ensemble_decision


app = FastAPI(title="Fake News Classifier")

class News(BaseModel):
    text: str

@app.post("/predict")
def predict(news: News):
    classical = predict_classical(news.text)
    bert_probs = predict_bert(news.text)
    result = ensemble_decision(classical, bert_probs)
    return result
