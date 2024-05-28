import numpy as np
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

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

import warnings
warnings.filterwarnings("ignore")

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
    word_patterns = doc[0]

    #lemmatize each word
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # create the bag of words array with 1, if word is found in current pattern
    for word in words:

        bag.append(1) if word in word_patterns else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

#shuffle features
random.shuffle(training)
training = np.array(training, dtype="object")

#create training and testing lists
# X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data is created")

#Train Model

#deep neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), 
                activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compiling model. SGD with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', 
              metrics=['accuracy'])

#Training and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, 
                 batch_size=5, verbose=1)
model.save('chatbot_model.keras', hist)


print("model is created")