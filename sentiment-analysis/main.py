# https://github.com/google-research/google-research/tree/master/goemotions/data/full_dataset
# Buradkai veri setiyle metin işleme yaparak. Gelen yorumdan o yorumdaki genel duygu tutumunu tahmin eden modeli geliştirelim.
# 1. fark => Sınıflandırma => 27 farklı duygu türü
import datasets

#split => Veriyi böl
data = datasets.load_dataset("go_emotions", split="train[:5000]")

texts = data['text']
labels = data['labels']

