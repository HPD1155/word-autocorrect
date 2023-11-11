import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import utils.preprocessing as preprocessing

df = pd.read_csv("data.csv")

# Load labels
X = df["word"]
y = df["predicted_word"]

def preprocessX(X):
    for i in range(len(X)):
        X[i] = preprocessing.Preprocessor.preprocess.characters(X[i])



preprocessX(X)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X)

print(X)

tokenizerY = Tokenizer()
tokenizerY.fit_on_texts(y)
y = tokenizerY.texts_to_sequences(y)
y = pad_sequences(y)
print(y)

model = models.load_model('model.keras')

while True:
    new_text = input("Enter word: ")
    new_text = preprocessing.Preprocessor.preprocess.characters(new_text)
    new_text = tokenizer.texts_to_sequences([new_text])
    new_text = pad_sequences(new_text, maxlen=X.shape[1])

    prediction = model.predict(new_text)
    prediction = prediction.tolist()
    predicted_label = np.argmax(prediction[0])
    predict_to_word = tokenizerY.sequences_to_texts([np.array([predicted_label])])

    print(predict_to_word[0])
    print("Confidence of word: " + str(prediction[0][predicted_label]))

    # Show vocab for y
    print(tokenizerY.word_index)

    plt.bar(range(len(prediction[0])), prediction[0])
    plt.show()