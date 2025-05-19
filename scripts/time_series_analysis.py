import pandas as pd
import matplotlib.pyplot as plt

def plot_daily_tweet_count(df):
    print("\nGünlük tweet sayısı hesaplanıyor...")

    df["date"] = pd.to_datetime(df["date"]).dt.date

    daily_counts = df.groupby("date").size()

    plt.figure(figsize=(14,6))
    plt.plot(daily_counts.index, daily_counts.values, marker="o", linestyle="-", color="blue", linewidth=2)
    plt.xlabel("Tarih")
    plt.ylabel("Tweet Sayısı")
    plt.title("Günlük Tweet Sayısı")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.figtext(0.5, -0.1,
                "Note: Veri setindeki tweetler kısa bir tarih aralığında toplandığı için grafik tekil bir yoğunluk göstermektedir.",
                wrap=True, horizontalalignment='center', fontsize=10)

    plt.savefig("visualizations/daily_tweet_count.png", bbox_inches="tight")
    plt.show()
