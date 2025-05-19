from transformers import pipeline
import pandas as pd

# Modelleri önceden yükle
print("Modeller yükleniyor...")
model1 = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
model2 = pipeline("zero-shot-classification", model="roberta-large-mnli", device=0)
model3 = pipeline("zero-shot-classification", model="microsoft/deberta-large-mnli", device=0)

models = [model1, model2, model3]
weights = [0.4, 0.3, 0.3]
candidate_labels = ["factual information", "misinformation", "unverified claim"]

def get_predictions(models, text, candidate_labels):
    predictions = []
    for model in models:
        result = model(text, candidate_labels)
        predictions.append(result["labels"][0])
    return predictions

def weighted_ensemble_predict(models, weights, text, candidate_labels):
    predictions = get_predictions(models, text, candidate_labels)
    vote_counts = {}
    for pred, weight in zip(predictions, weights):
        if pred not in vote_counts:
            vote_counts[pred] = 0
        vote_counts[pred] += weight
    return max(vote_counts.items(), key=lambda x: x[1])[0]

def ensemble_prediction(text):
    if not isinstance(text, str) or not text.strip():
        return "LABEL_UNKNOWN"
    pred = weighted_ensemble_predict(models, weights, text, candidate_labels)
    if "misinformation" in pred:
        return "misinformation"
    elif "factual information" in pred:
        return "true information"
    else:
        return "unverified claim"

def detect_misinformation(df):
    print("\n🔍 Ensemble modellerle yanlış bilgi tespiti yapılıyor...")
    df["misinformation_label"] = df["cleaned_text"].apply(ensemble_prediction)
    print("\n✅ Yanlış bilgi tespiti tamamlandı. İlk 5 sonuç:")
    print(df[["cleaned_text", "misinformation_label"]].head())
    return df
