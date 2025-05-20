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
    vote_counts = {}
    for model, weight in zip(models, weights):
        result = model(text, candidate_labels)
        for label, score in zip(result["labels"], result["scores"]):
            vote_counts[label] = vote_counts.get(label, 0) + score * weight
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

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_misinformation_detection(df):
    """
    Doğru ve tahmin edilen etiketlere göre sınıflandırma performansını değerlendirir.
    """
    if "label" not in df.columns or "misinformation_label" not in df.columns:
        print("⚠️ 'label' ve 'misinformation_label' sütunları bulunamadı.")
        return

    y_true = df["label"]
    y_pred = df["misinformation_label"]

    print("\n📋 Sınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, digits=3))

    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Doğruluk (Accuracy): {acc:.2%}")

    # Confusion matrix görselleştirme
    cm = confusion_matrix(y_true, y_pred, labels=["misinformation", "true information", "unverified claim"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["misinformation", "true information", "unverified claim"],
                yticklabels=["misinformation", "true information", "unverified claim"])
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.show()

