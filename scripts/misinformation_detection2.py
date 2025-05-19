from transformers import pipeline
import pandas as pd

def detect_misinformation(df):
    print("\nHugging Face modeli ile yanlış bilgi tespiti yapılıyor...")

    model_name = "facebook/bart-large-mnli"
    classifier = pipeline("zero-shot-classification", model=model_name, device=0)

    misinformation_labels = []

    candidate_labels = [
        "factual information", "misinformation", "unverified claim"
    ]

    for text in df["cleaned_text"]:
        if not text.strip():
            misinformation_labels.append("LABEL_UNKNOWN")
            continue

        prediction = classifier(text, candidate_labels)
        top_label = prediction["labels"][0]  # en yüksek skorlu etiket

    
        if "misinformation" in top_label:
            label = "misinformation"
        elif "factual information" in top_label:
            label = "true information"
        else:
            label = "unverified claim"

        misinformation_labels.append(label)

    df["misinformation_label"] = misinformation_labels

    print("\nYanlış bilgi tespiti tamamlandı. İlk 5 sonuç:")
    print(df[["cleaned_text", "misinformation_label"]].head())

    return df
