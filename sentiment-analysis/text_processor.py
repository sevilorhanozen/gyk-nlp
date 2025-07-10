import re
import nltk
import random 
nltk.download("stopwords")

stopwords = nltk.corpus.stopwords.words('english')

def process_text(text):
    text = clean_text(text)
    text = remove_mention(text)
    text = remove_stopwords(text)
    return text.strip()

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

def augment_text(text):
    words = text.split()
    random.shuffle(words)   
    return " ".join(words) 

# Data Augmentation

#Random Shuffle Data Augmentation

# I Feel very sad => sad label
# feel I very sad => sad label
# sad very I feel => sad label

# Synonym Replacement 
# Random Insertion
# Random Swap
# Random Deletion
# EDA
# Back Translation
