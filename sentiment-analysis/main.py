# https://github.com/google-research/google-research/tree/master/goemotions/data/full_dataset
# Buradkai veri setiyle metin işleme yaparak. Gelen yorumdan o yorumdaki genel duygu tutumunu tahmin eden modeli geliştirelim.
# 1. fark => Sınıflandırma => 27 farklı duygu türü
import datasets
import text_processor

#split => Veriyi böl
data = datasets.load_dataset("go_emotions", split="train[:5000]")

texts = data['text']
labels = data['labels']

clean_texts = [text_processor.process_text(text) for text in texts]
augmented_texts = [text_processor.augment_text(text) for text in clean_texts]

print(texts[0])
print(clean_texts[0])
print(augmented_texts[0])

final_texts = clean_texts + augmented_texts
final_labels = list(labels) + list(labels)

print(len(final_texts))
print(len(final_labels))

print(final_texts[0])
print(final_labels[:50])

# Eğer labelimiz "sad,happy,love" gibi bir text label ise. [sad,happy,love,angry] => Encoding

# sad => 0
# happy => 1

# Eğer birden fazlaa label varsa bir veri için. "sad","angry"
# MultiLabelBinarizer
# OneHotEncoder
# [6,9,27] => 10
import pickle 
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(final_labels)
num_classes = y.shape[1]

with open("mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

# Label kısmını hallettim. Derin öğrenmeye girmeye hazır.
# Text kısmını halletmeliyim.

# Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(final_texts)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

sequences = tokenizer.texts_to_sequences(final_texts)
X = pad_sequences(sequences, padding="post", maxlen=100)

print(X[:50])
#

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
print("Durdur")
model = Sequential(
    [
        layers.Embedding(input_dim=10000, output_dim=128, input_length=100),
        layers.SpatialDropout1D(0.5), # Dropout => Embedding çıkışlarında kullanılır. (Bir cümleyi tamamen kullanmadan dropout yapar.)
        # Gated Recurrent Unit
        # Update Gate -> Önceki bilgiyi ne kadar tutacağımı belirliyor.
        # Reset Gate -> Önceki bilgiyi ne kadar silineceğini belirliyor.
        # LSTM'e göre Daha az parametre ve daha hızlı.
        # Forget Gate, -> Geçmişten gelen hangi bilgileri unutacağını belirler
        # Input Gate, -> Yeni gelen bilgiden ne kadarını belleğe ekleyeceğini belirler.
        # Output Gate -> Bellekteki bilginin ne kadarının çıktıya yansıyacağını belirler.
        layers.Bidirectional(layers.GRU(128, return_sequences=True)),
        layers.LayerNormalization(),  # Katman çıktılarını normalize eder. (Öğrenmeyi daha stabil hale getirir.)
        layers.Bidirectional(layers.GRU(64, return_sequences=False)),
        # Öğrenilmiş bilgileri alıp 64 farklı karar mekanizmasından geçir.
        layers.Dense(64, activation="relu"),
        # Dense katmanındaki karar mekanizmalarının %50'sini rastgele kapat. Ezberlemeyi önle.
        layers.Dropout(0.5),
        # Output Layer
        layers.Dense(num_classes, activation="sigmoid") # Toplamı 1 olacak şekilde her bir duyguya oran verir. 
        # 0.1 => sad
        # 0.005 => happy
        # 0.85 => angry
        # 0.045 => love
    ]
)
import tensorflow as tf
# binary_crossentropy => Çoklu etiket için uygun bir loss function.
# Area Under The Curve => ROC eğrisi altındaki alanı hesaplar.
# AUC => 0-1 arasında değer alır 1 => mükemmel 0.5 ve altı ise kötü.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

model.summary()

model.fit(X,y, epochs=5,  batch_size=64, validation_split=0.2)

model.save("sentiment_analysis_model.h5")

# 20:35