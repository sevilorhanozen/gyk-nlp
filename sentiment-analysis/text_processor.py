import re
import nltk

nltk.download("stopwords")

stopwords = nltk.corpus.stopwords.words('english')

def process(text):
    pass

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text) # metindeki linkleri kaldır
    text = re.sub(r'\d+', '', text) # Sayıları kaldır
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # Noktalama işaretlerini kaldır
    return text
def remove_mention(text):
    text = re.sub(r"@\S+", "", text) # metindeki mentionleri kaldır
    return text

def remove_stopwords(text):
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text