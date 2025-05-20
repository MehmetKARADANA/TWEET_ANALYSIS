from scripts import misinformation_dedection, read_data, clean_data, save_clean_data, sentiment_analysis, plot_sentiment_distribution,  topic_modeling, time_series_analysis,  network_analysis, topic_keywords_analysis, misinformation_analysis


df = read_data.load_data("data/test_data.csv")

"""
df = read_data.load_data("data/covid19_tweets.csv")
df = read_data.load_data("analyzed_data/misinformation_tweets.csv")
"""
df = clean_data.clean_tweets(df)

df = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)


save_clean_data.save_clean_data(df, "cleaned_data/cleaned_tweets.csv")

df = sentiment_analysis.analyze_sentiments(df)

plot_sentiment_distribution.plot_sentiment_distribution(df)

df = topic_modeling.model_topics(df)

df = df[df["topic"] != -1]
print(f"\n-1 ID'li tweetler çıkarıldı. Kalan tweet sayısı: {len(df)}")

time_series_analysis.plot_daily_tweet_count(df)

topic_keywords_analysis.analyze_topic_keywords(df)

network_analysis.build_user_mention_network(df)

df = misinformation_dedection.detect_misinformation(df)

misinformation_analysis.analyze_misinformation(df)

misinformation_dedection.evaluate_misinformation_detection(df)