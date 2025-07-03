import pandas as pd
import re
import string

df = pd.read_csv('spam.csv', encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam':1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) # sayıları temizle
    text = text.translate(str.maketrans('','', string.punctuation)) # noktalama işaretlerini temizle
    return text

df['clean_text'] = df['text'].apply(clean_text)
print(df.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)


# Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

# num_words => Kelime sayısı
# out-of-vocabulary token => <OOV>
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

print(tokenizer.word_index.items())

# Metinleri sayıya çevir (sequence)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# İlk 3 orijinal mesaj
for i in range(3):
    print(f"{i+1}. Metin: {X_train.iloc[i]}")
# İlk 3 sayısal mesaj
for i in range(3):
    print(f"{i+1}. Metin: {X_train_seq[i]}")

# Padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Bütün cümleleri belirlenen uzunluğa getir.
max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
#
for i in range(3):
    print(f"{i+1}. Metin: {X_train_pad[i]}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Embedding Layer Detayları

model = Sequential()
# input_dim => Toplam kelime sayısı
# output_dim => Vektör boyutu (Her idyi kaç boyutlu vektörle temsil edeceğiz)
# input_length => Her mesajın kaç kelimelik olacağı
model.add(Embedding(input_dim=5000, output_dim=16, input_length=max_length)) # Kelimeleri sayılardan oluşan anlamlı vektörlere çevir.
# 42 -> [0.25,0.13,-0.27, 0.5]
# Kelimelere id atamak (Tokenizer)
# 42 gibi sayılar model için anlamlı değil.. 
# ID => Anlamlı bir uzayda vektörel temsil edeyim.

# Long Short Term Memory (LSTM) Layer
# RNN => Recurrent Neural Network
model.add(LSTM(64)) # 64 => Nöron sayısı
# Bağlamı hatırlamak için.
# Ali sabah okula gitti. Akşam eve geldi.

# Forget Gate, Input Gate, Output Gate (ARAŞTIRMA)
# LSTM sırayla gelen verileri işlerken geçmişte gördüğü önemli bilgileri hatırlayan bir yapay zeka katmanı.

# Activation Function nedir?

# Dense -> Fully Connected Layer
# 1 => 1 tane nöron çünkü binary classification -> spam değil mi?
model.add(Dense(1, activation="sigmoid")) # sigmoid => 0-1 arası değerler verir. %75 spam %25 ham 0.75

# loss => Her tahminde ne kadar hata yapıldığını ölçer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()
# Epoch => veriyi baştan sona kaç kere göreyim?
history = model.fit(X_train_pad, y_train, epochs=10, validation_data=(X_test_pad, y_test))

# Eğitim sonunda modeli kaydet.
model.save("sms_model.h5")
#model.save("sms_model2.keras") 

import pickle

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)