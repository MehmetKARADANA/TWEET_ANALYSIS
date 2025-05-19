from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiments(df):
   
    print("\nDuygu analizi yapılıyor...")
    
   
    df["sentiment"] = df["cleaned_text"].apply(lambda x: sentiment_pipeline(x)[0]["label"])

    print("Duygu analizi tamamlandı. İlk 5 sonuç:")
    print(df[["cleaned_text", "sentiment"]].head())

    return df
