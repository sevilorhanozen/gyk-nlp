#Modeli kullan.

from tensorflow.keras.models import load_model
import pickle
import string 
import re

MODEL_PATH = "sms_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# Dosyadan modeli yükle.
model = load_model(MODEL_PATH)

# Dosyadan tokenizer'ı yükle.
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) # sayıları temizle
    text = text.translate(str.maketrans('','', string.punctuation)) # noktalama işaretlerini temizle
    return text

from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict(message):
    message = clean_text(message)
    seq = tokenizer.texts_to_sequences([message])
    pad = pad_sequences(seq, maxlen=100)
    prediction = model.predict(pad)
    print(f"Tahmin Oranı:  {prediction[0][0]:.4f}")
    if prediction[0][0] > 0.5:
        print("Spam")
    else:
        print("Ham")

#predict("You have won a free iPhone 13 Pro Max! Click the link to claim your prize.")
#predict("I'm sorry to hear that you're having trouble with your account. Let me know if I can help you with anything.")
if __name__ == "__main__":
    text = input("Bir sms giriniz:")
    predict(text)
