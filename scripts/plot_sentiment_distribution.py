import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df):
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x="sentiment", data=df, palette="coolwarm")
    plt.title("Tweetlerin Duygu Dağılımı")
    plt.xlabel("Duygu Türü")
    plt.ylabel("Tweet Sayısı")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig("visualizations/sentiment_distribution.png")
    plt.show()
