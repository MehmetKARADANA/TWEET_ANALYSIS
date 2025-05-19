import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import os
from collections import Counter
import numpy as np

def analyze_misinformation(df):
   
    print("\nYanlış bilgi içerdiği tahmin edilen tweetler analiz ediliyor...")

    output_dir = "analyzed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    misinfo_df = df[df["misinformation_label"] == "misinformation"]
    true_info_df = df[df["misinformation_label"] == "true information"]
    unverified_df = df[df["misinformation_label"] == "unverified claim"]

    misinfo_df.to_csv(f"{output_dir}/misinformation_tweets.csv", index=False)
    true_info_df.to_csv(f"{output_dir}/true_information_tweets.csv", index=False)
    unverified_df.to_csv(f"{output_dir}/unverified_tweets.csv", index=False)

    print(f"Toplam yanlış bilgi içerdiği belirlenen tweet sayısı: {len(misinfo_df)}")
    print(f"Toplam doğru bilgi içerdiği belirlenen tweet sayısı: {len(true_info_df)}")
    print(f"Toplam doğrulanmamış iddia içeren tweet sayısı: {len(unverified_df)}")

    plt.figure(figsize=(8,6))
    labels = ['Yanlış Bilgi', 'Doğru Bilgi', 'Doğrulanmamış']
    counts = [len(misinfo_df), len(true_info_df), len(unverified_df)]
    
    non_zero_counts = [c for c in counts if c > 0]
    non_zero_labels = [l for l, c in zip(labels, counts) if c > 0]
    
    if non_zero_counts:
        plt.pie(non_zero_counts, labels=non_zero_labels, autopct='%1.1f%%', 
                colors=['red', 'green', 'yellow'][:len(non_zero_counts)])
        plt.title('Tweet Etiket Dağılımı')
        plt.savefig("visualizations/misinformation_label_distribution.png", bbox_inches="tight")
        plt.close()
    else:
        print("Uyarı: Hiç tweet bulunamadı, etiket dağılım grafiği oluşturulamadı.")

    for label, df_label in [('Yanlış Bilgi', misinfo_df), ('Doğru Bilgi', true_info_df), ('Doğrulanmamış', unverified_df)]:
        if not df_label.empty:
            df_label["date"] = pd.to_datetime(df_label["date"]).dt.date
            daily_counts = df_label.groupby("date").size()
            
            plt.figure(figsize=(14,6))
            plt.plot(daily_counts.index, daily_counts.values, marker="o", linestyle="-", linewidth=2)
            plt.xlabel("Tarih")
            plt.ylabel("Tweet Sayısı")
            plt.title(f"{label} İçeren Tweetlerin Günlük Yayılımı")
            plt.xticks(rotation=45)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"visualizations/{label.lower().replace(' ', '_')}_daily_spread.png", bbox_inches="tight")
            plt.close()


    if not df.empty and 'sentiment' in df.columns:
        plt.figure(figsize=(10,6))
        sentiment_data = pd.DataFrame({
            'Yanlış Bilgi': misinfo_df['sentiment'].value_counts(),
            'Doğru Bilgi': true_info_df['sentiment'].value_counts(),
            'Doğrulanmamış': unverified_df['sentiment'].value_counts()
        }).fillna(0)
        sentiment_data.plot(kind='bar')
        plt.title('Etiketlere Göre Duygu Dağılımı')
        plt.xlabel('Duygu')
        plt.ylabel('Tweet Sayısı')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("visualizations/sentiment_comparison.png", bbox_inches="tight")
        plt.close()

    if not df.empty and 'topic' in df.columns:
        plt.figure(figsize=(12,6))
        topic_data = pd.DataFrame({
            'Yanlış Bilgi': misinfo_df['topic'].value_counts(),
            'Doğru Bilgi': true_info_df['topic'].value_counts(),
            'Doğrulanmamış': unverified_df['topic'].value_counts()
        }).fillna(0)
        topic_data.plot(kind='bar')
        plt.title('Etiketlere Göre Konu Dağılımı')
        plt.xlabel('Konu')
        plt.ylabel('Tweet Sayısı')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("visualizations/topic_comparison.png", bbox_inches="tight")
        plt.close()

    if not df.empty and 'retweet_count' in df.columns and 'favorite_count' in df.columns:
        interaction_data = pd.DataFrame({
            'Yanlış Bilgi': [misinfo_df['retweet_count'].mean(), misinfo_df['favorite_count'].mean()],
            'Doğru Bilgi': [true_info_df['retweet_count'].mean(), true_info_df['favorite_count'].mean()],
            'Doğrulanmamış': [unverified_df['retweet_count'].mean(), unverified_df['favorite_count'].mean()]
        }, index=['Ortalama Retweet', 'Ortalama Beğeni'])
        
        plt.figure(figsize=(10,6))
        interaction_data.plot(kind='bar')
        plt.title('Etiketlere Göre Ortalama Etkileşim')
        plt.ylabel('Ortalama Sayı')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("visualizations/interaction_comparison.png", bbox_inches="tight")
        plt.close()

    print("Tüm analiz grafikleri kaydedildi.")
    print(f"Tweetler {output_dir} klasörüne etiketlerine göre ayrı ayrı kaydedildi.")
