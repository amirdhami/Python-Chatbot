import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('chatbot_model.h5')

import json
import random

import warnings
warnings.filterwarnings("ignore")

#Load Pickle
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array

    #sentence_words = nltk.word_tokenize(sentence)

    # stemming every word - reducing to base form
    #sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return #sentence_words