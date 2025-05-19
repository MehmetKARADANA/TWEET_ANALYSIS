from utils import text_cleaning

def clean_tweets(df):
   
    print("\nTweetler temizleniyor...")
    df["cleaned_text"] = df["text"].apply(text_cleaning.clean_text)
    print("Temizleme tamamlandı. Temizlenmiş ilk 5 tweet:")
    print(df[["text", "cleaned_text"]].head())
    return df
