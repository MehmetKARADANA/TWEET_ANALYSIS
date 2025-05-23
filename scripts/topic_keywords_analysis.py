from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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

        if not common_words:
            print(f"⚠️ No words found for Topic {topic}. Skipping...")
            continue
      
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(common_words))

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topic {topic}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"visualizations/topic_keywords/wordclouds/topic_{topic}_wordcloud.png")
        plt.close()

    print("Word clouds saved in'visualizations/topic_keywords'")
