import pandas as pd
import re
import string
# encoding => Dosyadaki harf,sayı (karakterlerin) byte dizisine dönüştürülme yöntemi.
df = pd.read_csv('spam.csv', encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam':1})

# Bütün verimi normalize etmek için. (Hepsi küçük harf, sayıları temizlenmiş, noktalama işaretleri temizlenmiş)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) # sayıları temizle
    text = text.translate(str.maketrans('','', string.punctuation)) # noktalama işaretlerini temizle
    return text

df['clean_text'] = df['text'].apply(clean_text)
print(df.shape)

# Makine Öğrenmesi -> train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# Vectorizer -> Metinleri sayısal değerlere dönüştürür.
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(vectorizer.get_feature_names_out())
print(X_train_vectorized.toarray())
# Kelimelerden numerik sözlük oluşturur.

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

# Benchmark
from sklearn.metrics import classification_report,accuracy_score

print(accuracy_score(y_test, y_pred))
print("*****")
print(classification_report(y_test, y_pred))


#new_sms = "You have won a free iPhone 13 Pro Max! Click the link to claim your prize."
new_sms = "I'm sorry to hear that you're having trouble with your account. Let me know if I can help you with anything."
new_sms = clean_text(new_sms)

new_sms_vectorized = vectorizer.transform([new_sms])

prediction = model.predict(new_sms_vectorized)

print("Tahmin: ", "SPAM" if prediction[0] == 1 else "SPAM DEĞİL")