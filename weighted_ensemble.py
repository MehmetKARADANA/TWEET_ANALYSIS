from transformers import pipeline
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Modeller yükleniyor...")
model1 = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
model2 = pipeline("zero-shot-classification", model="roberta-large-mnli", device=0)
model3 = pipeline("zero-shot-classification", model="microsoft/deberta-large-mnli", device=0)

models = [model1, model2, model3]
weights = [0.4, 0.3, 0.3]  # Ağırlıklar toplamı 1

candidate_labels = [
    "factual information", "misinformation", "unverified claim"
]

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
    print("\nEnsemble modellerle yanlış bilgi tespiti yapılıyor...")
    
    # Batch boyutu
    BATCH_SIZE = 32
    
    # Batch'ler halinde işle
    all_predictions = []
    total_batches = len(df) // BATCH_SIZE + (1 if len(df) % BATCH_SIZE != 0 else 0)
    
    for i in range(0, len(df), BATCH_SIZE):
        batch = df["cleaned_text"].iloc[i:i+BATCH_SIZE]
        batch_predictions = []
        
        for text in batch:
            if not isinstance(text, str) or not text.strip():
                batch_predictions.append("LABEL_UNKNOWN")
                continue
                
            pred = weighted_ensemble_predict(models, weights, text, candidate_labels)
            
            if "misinformation" in pred:
                label = "misinformation"
            elif "factual information" in pred:
                label = "true information"
            else:
                label = "unverified claim"
                
            batch_predictions.append(label)
            
        all_predictions.extend(batch_predictions)
        print(f"\rİşlenen batch: {i//BATCH_SIZE + 1}/{total_batches}", end="")
    
    print("\nTahminler tamamlandı.")
    df["misinformation_label"] = all_predictions
    print("\nİlk 5 örnek:")
    print(df[["cleaned_text", "misinformation_label"]].head())
    return df

def evaluate_models(df):
    print("\n📊 Modeller değerlendiriliyor...")

    if "cleaned_text" not in df.columns or "label" not in df.columns:
        print("Hata: 'cleaned_text' ve 'label' sütunları eksik!")
        print("Mevcut sütunlar:", df.columns.tolist())
        return

    # 1. Ensemble Model Değerlendirmesi
    print("\n🔍 Ensemble Model Değerlendirmesi:")
    predicted_labels = df["misinformation_label"]
    true_labels = df["label"]

    # Doğruluk oranını hesapla
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\n📈 Ensemble Model Doğruluk Oranı: {accuracy:.2%}")

    print("\n📋 Ensemble Model Sınıflandırma Raporu:")
    print(classification_report(true_labels, predicted_labels, digits=3))

    # 2. Duygu Analizi Modeli Değerlendirmesi
    print("\n😊 Duygu Analizi Modeli Değerlendirmesi:")
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    # Eğer veri setinde duygu etiketleri varsa
    if "sentiment" in df.columns:
        predicted_sentiments = []
        for text in df["cleaned_text"]:
            result = sentiment_analyzer(text)[0]
            predicted_sentiments.append(result["label"])

        print("\n📋 Duygu Analizi Sınıflandırma Raporu:")
        print(classification_report(df["sentiment"], predicted_sentiments, digits=3))

    # 3. Konu Modelleme Değerlendirmesi
    print("\n📚 Konu Modelleme Değerlendirmesi:")
    from bertopic import BERTopic
    topic_model = BERTopic(language="english")
    
    # Konu modelleme için metinleri dönüştür
    topics, probabilities = topic_model.fit_transform(df["cleaned_text"])
    
    # Konu dağılımını görselleştir
    plt.figure(figsize=(12, 6))
    sns.countplot(x=topics, palette="viridis")
    plt.title("Konu Dağılımı")
    plt.xlabel("Konu ID")
    plt.ylabel("Tweet Sayısı")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Görselleştirmeyi kaydet
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    plt.savefig("visualizations/topic_distribution.png")
    plt.close()

    # Konu kalitesi metrikleri
    topic_info = topic_model.get_topic_info()
    print("\n📊 Konu Modelleme Metrikleri:")
    print(f"Toplam Konu Sayısı: {len(topic_info)}")
    print(f"Ortalama Konu Boyutu: {topic_info['Count'].mean():.2f}")
    print(f"En Büyük Konu Boyutu: {topic_info['Count'].max()}")
    print(f"En Küçük Konu Boyutu: {topic_info['Count'].min()}")

    print("\n✅ Tüm modeller değerlendirildi ve sonuçlar kaydedildi.")

if __name__ == "__main__":
    # CSV dosyasını daha esnek bir şekilde oku
    df = pd.read_csv("data/test_dataset_1000.csv", 
                     encoding="windows-1254",
                     quoting=1,  # QUOTE_ALL modu
                     escapechar='\\')  # Kaçış karakteri
    
    # Sütun isimlerini düzelt
    df.columns = ['tweet_id', 'cleaned_text', 'label']
    
    df = detect_misinformation(df)
    evaluate_models(df)
    df.to_csv("results/ensemble_predictions.csv", index=False)

