import pandas as pd

def load_data(filepath):

    df = pd.read_csv(filepath)

    selected_columns = [
        "user_name", "user_location", "user_description",
        "user_created", "user_followers", "user_friends",
        "user_favourites", "user_verified", "date",
        "text", "hashtags", "source", "is_retweet",
        "cleaned_text", "sentiment", "topic", "label"
    ]
    
    if all(col in df.columns for col in selected_columns):
        df = df[selected_columns]
    else:
        missing = [col for col in selected_columns if col not in df.columns]
        raise ValueError(f"Veri setinde şu sütunlar eksik: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")


    print(f"Veri seti başarıyla yüklendi. {df.shape[0]} satır, {df.shape[1]} sütun.")
    print("\nİlk 5 satır:\n", df.head())

    return df
