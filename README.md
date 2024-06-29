# Python Chatbot

This repository contains a Python-based chatbot that uses a neural network to understand and respond to user queries. The chatbot can handle a variety of intents such as greetings, goodbyes, and specific tasks related to healthcare.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Training the Chatbot](#training-the-chatbot)
- [Running the Chatbot](#running-the-chatbot)
- [Dependencies](#dependencies)

## Installation

To get started with the Python Chatbot, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/python-chatbot.git
   
2. cd into chatbot directory:
   ```sh
   cd <project_directory>
   
3. Install requirements:
   ```sh
   pip install -r requirements.txt

## Usage
### Files:
- intents.json: This file contains the training data for the chatbot. It includes different tags, patterns, and responses.
- train_chatbot.py: This script is used to train the chatbot model.
- gui_chatbot.py: This script is used to run the chatbot with a graphical user interface.
- chatbot_model.h5, classes.pkl, words.pkl, chatbot_model.keras: These files are generated after training the chatbot and are used for the model's predictions.

### Training the Chatbot:

To train the chatbot model, execute the following command:

```sh
   python train_chatbot.py
```
This will process the intents.json file, train a neural network, and save the model along with necessary metadata files (words.pkl, classes.pkl, chatbot_model.h5, and chatbot_model.keras).

### Running the Chabot

To run the chatbot with a graphical user interface, execute:

```sh
   python gui_chatbot.py
```
This will launch a simple GUI where you can interact with the chatbot.

## Dependencies

Ensure you have the following dependencies installed. 
- TensorFlow
- Keras
- Numpy
- Pickle
- NLTK
- Tkinter (for GUI)
- Note: Make sure that when you import the various Keras modules, you import them from the same library. In this project, I've imported all Keras dependencies from tensorflow.keras. You may run into errors otherwise.

For any issues or contributions, please feel free to create a pull request. Thank you!
