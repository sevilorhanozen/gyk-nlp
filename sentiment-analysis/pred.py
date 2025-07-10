import tensorflow as tf
import pickle


model = tf.keras.models.load_model("sentiment_analysis_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

from datasets import load_dataset

data = load_dataset("go_emotions", split="test[:1000]")
texts = data['text']
labels = data['labels']
label_names = data.features['labels'].feature.names
print(label_names)

# ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

print(texts[:5])
print(labels[:5])

from tensorflow.keras.preprocessing.sequence import pad_sequences

X_test = tokenizer.texts_to_sequences(texts)
X_test = pad_sequences(X_test, padding="post", maxlen=100)

y_test = mlb.transform(labels)

loss, accuracy, auc = model.evaluate(X_test, y_test)

print(f"Loss: {loss}, Accuracy: {accuracy}, AUC: {auc}")

test_texts = ["I am very sad", "I am very sad", "I am very angry"]

test_sequences = tokenizer.texts_to_sequences(test_texts)
test_sequences = pad_sequences(test_sequences, padding="post", maxlen=100)

y_pred = model.predict(test_sequences)

print("y_pred", y_pred)

import numpy as np

predicted_indices = np.where(y_pred[0] > 0.5)[0]
predicted_labels = [mlb.classes_[i] for i in predicted_indices]


print(predicted_labels)
print(data)
print(label_names[predicted_labels[0]])


