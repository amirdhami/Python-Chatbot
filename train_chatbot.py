import numpy as np
from tensorflow import keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

intents_file = open('intents.json').read()
intents = json.loads(intents_file)

#Preprocess Data

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        #add documents in corpus
        documents.append((word, intent['tag']))

        #add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Lemmaztize and lower each word and remove duplicates

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

#sort classes
classes = sorted(list(set(classes)))

#documents = combom btw. patterns and intents
print(len(documents), "documents")

#classes = intents
print(len(classes), "classes", classes)

#words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Create training data

training = []

#create empty array for output
output_empty = [0] * len(classes)

#training set, bag of words for every sentence
for doc in documents:
    #init bag of words
    bag = []

    #list of tokenized words for pattern
    word_pattenrs = doc[0]

