def ensemble_decision(classical_preds, bert_probs, threshold=0.65):
    """
    classical_preds: {"svm": 0/1, "lr": 0/1}
    bert_probs: [P_FAKE, P_REAL]
    """

    p_fake, p_real = bert_probs

    # Trust BERT only if confident
    if p_real >= threshold:
        return {"label": "REAL", "confidence": round(p_real, 3)}

    if p_fake >= threshold:
        return {"label": "FAKE", "confidence": round(p_fake, 3)}

    # Otherwise fallback to classical models
    votes = classical_preds["svm"] + classical_preds["lr"]

    if votes >= 1:
        return {"label": "REAL", "confidence": 0.99}
    else:
        return {"label": "FAKE", "confidence": 0.99}
