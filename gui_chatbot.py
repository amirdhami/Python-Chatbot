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