# NLP nedir?

# NLP ile neler yapıyoruz? => 
# Metin Sınıflandırma -> E posta spam mı değil mi?
# Duygu Analizi -> (mutlu mu ? üzgün mü?)
# Özetleme -> 
# Metin Üretimi -> 
# Chatbot 
# Named Entity Recognition



# Bölüm 1 Müfredat Konuları


# Kütüphaneler -> numpy/pandas 
# NLTK => Temel nlp işlemleri yapan.
# scikit-learn


import nltk 
nltk.download('punkt_tab') # punkt_tab => Tokenizer

text = "Natural Language Processing is a branch of artificial intelligence."

# Tokenization
from nltk.tokenize import word_tokenize

tokens = word_tokenize(text)
print(tokens)
#

# Stop-Word Removal
# is,the,on,at,in
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) #Dosyadaki kelimeleri oku.
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens)
#

# Lemmatization -> Kök haline getirme
# running -> run
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
# v =>verb -> fiil
# n =>noun -> isim
# a => adjective -> sıfat
# r => adverb (zarf)
print(lemmatizer.lemmatize('running', pos='n'))
# He is running.
# He went to running.

# --
# He is running every day.
# I run every day.
# She ran yesterday.
# --

#
# He be run every day
# I run every day
# She run yesterday
#

# injured -> injure






#







