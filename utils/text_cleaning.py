import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words("english"))

def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

def clean_text(text):
    if isinstance(text, str):
        text = remove_emojis(text)
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = word_tokenize(text)
        text = [word for word in text if word not in stop_words]
        text = " ".join(dict.fromkeys(text))
        return text
    return ""
