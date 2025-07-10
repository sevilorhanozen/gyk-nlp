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
print(final_labels[0])

# Eğer labelimiz "sad,happy,love" gibi bir text label ise. [sad,happy,love,angry] => Encoding

# sad => 0
# happy => 1

# Eğer birden fazlaa label varsa bir veri için. "sad","angry"
# MultiLabelBinarizer
# OneHotEncoder

# 19:40