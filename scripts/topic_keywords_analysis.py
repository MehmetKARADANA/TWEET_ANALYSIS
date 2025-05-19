from collections import Counter
import os

def analyze_topic_keywords(df, top_n=10):
    print("\nKonu bazlı anahtar kelime analizi yapılıyor...")

    if not os.path.exists("visualizations/topic_keywords"):
        os.makedirs("visualizations/topic_keywords")

    topics = df["topic"].unique()

    for topic in topics:
        topic_df = df[df["topic"] == topic]
        all_words = " ".join(topic_df["cleaned_text"]).split()

        word_counts = Counter(all_words)
        common_words = word_counts.most_common(top_n)
        with open(f"visualizations/topic_keywords/topic_{topic}_keywords.txt", "w", encoding="utf-8") as f:
            f.write(f"Topic {topic} - En Sık {top_n} Kelime:\n")
            for word, freq in common_words:
                f.write(f"{word}: {freq}\n")

    print(f"Anahtar kelimeler visualizations/topic_keywords klasörüne kaydedildi.")
