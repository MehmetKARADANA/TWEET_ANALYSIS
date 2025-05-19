import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import pipeline
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models():
    print("📊 Modeller değerlendiriliyor...")

    data_path = "data/test_dataset_1000.csv"

    if not os.path.exists(data_path):
        print(f"Hata: Veri dosyası bulunamadı -> {data_path}")
        return

    # CSV dosyasını doğru şekilde oku
    df = pd.read_csv(data_path, encoding="windows-1254")

    if "cleaned_text" not in df.columns or "label" not in df.columns:
        print("Hata: 'cleaned_text' ve 'label' sütunları eksik!")
        print("Mevcut sütunlar:", df.columns.tolist())
        return

    # 1. Yanlış Bilgi Tespiti Modeli Değerlendirmesi
    print("\n🔍 Yanlış Bilgi Tespiti Modeli Değerlendirmesi:")
    misinformation_classifier = pipeline("zero-shot-classification", 
                                      model="facebook/bart-large-mnli", 
                                      device=0)

    candidate_labels = ["factual information", "misinformation", "unverified claim"]
    predicted_labels = []

    for text in df["cleaned_text"]:
        if not text.strip():
            predicted_labels.append("LABEL_UNKNOWN")
            continue

        prediction = misinformation_classifier(text, candidate_labels)
        top_label = prediction["labels"][0]

        if "misinformation" in top_label:
            label = "misinformation"
        elif "factual information" in top_label:
            label = "true information"
        else:
            label = "unverified claim"

        predicted_labels.append(label)

    print("\n📋 Yanlış Bilgi Tespiti Sınıflandırma Raporu:")
    print(classification_report(df["label"], predicted_labels, digits=3))

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
    evaluate_models()