from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns

def model_topics(df):
    
    print("\nBERTopic modeli ile konu analizi yapılıyor...")

    topic_model = BERTopic(language="english")

    topics, probabilities = topic_model.fit_transform(df["cleaned_text"])

    df["topic"] = topics

    print("Konu modelleme tamamlandı. İlk 5 konu:")
    print(df[["cleaned_text", "topic"]].head())

    plt.figure(figsize=(10,6))
    sns.countplot(x="topic", data=df, palette="coolwarm")
    plt.title("Konu Dağılımı")
    plt.xlabel("Konu ID")
    plt.ylabel("Tweet Sayısı")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig("visualizations/topic_distribution.png")
    plt.show()

    return df
