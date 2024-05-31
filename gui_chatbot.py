import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('chatbot_model.h5')

import json
import random

# Creating tkinter GUI
import tkinter
from tkinter import *

# Load Pickle files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    """
    Tokenize the input sentence and lemmatize each word to its base form.

    Args:
        sentence (str): The input sentence to be processed.

    Returns:
        list: A list of lemmatized words from the input sentence.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_details=True):
    """
    Convert the input sentence into a bag-of-words array.

    Args:
        sentence (str): The input sentence.
        words (list): The list of known words.
        show_details (bool): Whether to print details of the process.

    Returns:
        numpy.array: The bag-of-words representation of the input sentence.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return np.array(bag)

def predict_class(sentence):
    """
    Predict the class of the input sentence based on the trained model.

    Args:
        sentence (str): The input sentence.

    Returns:
        list: A list of predicted class intents with their probabilities.
    """
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def getResponse(ints, intents_json):
    """
    Get a random response based on the predicted intent.

    Args:
        ints (list): The list of predicted intents.
        intents_json (dict): The JSON object containing intents and responses.

    Returns:
        str: A randomly selected response based on the predicted intent.
    """
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()

    EntryBox.delete("0.0",END)


    if msg != '':

        ChatBox.config(state=NORMAL)

        ChatBox.insert(END, "You: " + msg + '\n\n')

        ChatBox.config(foreground="#446665", font=("Verdana", 12 )) 


        ints = predict_class(msg)

        res = getResponse(ints, intents)

        

        ChatBox.insert(END, "Bot: " + res + '\n\n')           


        ChatBox.config(state=DISABLED)

        ChatBox.yview(END)