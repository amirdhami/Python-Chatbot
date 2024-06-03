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
    """
    Handles the sending of a user message from the input box to the chatbot,
    and displays the chatbot's response in the chat box.

     Args:
        None

    Returns: 
        None
    """
    # Get the user message from the EntryBox, stripping any extra spaces
    msg = EntryBox.get("1.0", 'end-1c').strip()

    # Clear the EntryBox after getting the message
    EntryBox.delete("0.0", END)

    if msg != '':
        # Enable the ChatBox to insert the user's message
        ChatBox.config(state=NORMAL)

        # Insert the user's message into the ChatBox
        ChatBox.insert(END, "You: " + msg + '\n\n')

        # Configure the appearance of the ChatBox
        ChatBox.config(foreground="#446665", font=("Verdana", 12))

        # Predict the class of the user's message
        ints = predict_class(msg)

        # Get the response from the chatbot based on the predicted class
        res = getResponse(ints, intents)

        # Insert the chatbot's response into the ChatBox
        ChatBox.insert(END, "Bot: " + res + '\n\n')

        # Disable the ChatBox to prevent user input directly into it
        ChatBox.config(state=DISABLED)

        # Scroll to the end of the ChatBox to show the latest message
        ChatBox.yview(END)

# Initialize the main window for the chatbot GUI
root = Tk()

# Set the title of the main window
root.title("Chatbot")

# Set the size of the main window
root.geometry("400x500")

# Prevent the main window from being resizable
root.resizable(width=FALSE, height=FALSE)

# Create the chat window (ChatBox) where messages will be displayed
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial")

# Disable the ChatBox to prevent user input directly into it
ChatBox.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
                    command= send )

#Create the box to enter message
EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial")
