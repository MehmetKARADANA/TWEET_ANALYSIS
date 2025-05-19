def save_clean_data(df, filepath="cleaned_data/cleaned_tweets.csv"):
    
    df.to_csv(filepath, index=False)
    print(f"Temizlenmiş veri cleaned_data klasörüne kaydedildi: {filepath}")
