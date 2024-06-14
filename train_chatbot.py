import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def load_intents(file_path):
    """
    Load intents from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing intents.

    Returns:
        dict: A dictionary of intents.
    """
    with open(file_path) as file:
        return json.loads(file.read())

def preprocess_data(intents):
    """
    Preprocess data by tokenizing, lemmatizing, and organizing into words, classes, and documents.

    Args:
        intents (dict): A dictionary of intents.

    Returns:
        tuple: A tuple containing words, classes, and documents.
    """
    words = []
    classes = []
    documents = []
    ignore_letters = ['!', '?', ',', '.']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            documents.append((word, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    return words, classes, documents

def create_training_data(words, classes, documents):
    """
    Create training data from words, classes, and documents.

    Args:
        words (list): A list of words.
        classes (list): A list of classes.
        documents (list): A list of documents.

    Returns:
        tuple: A tuple containing training data, train_x, and train_y.
    """
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype="object")

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    return training, train_x, train_y

def build_model(input_shape, output_shape):
    """
    Build and compile a deep neural network model.

    Args:
        input_shape (int): The shape of the input data.
        output_shape (int): The shape of the output data.

    Returns:
        keras.Sequential: A compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_and_save_model(model, train_x, train_y, epochs, batch_size, model_path):
    """
    Train the model and save it to a file.

    Args:
        model (keras.Sequential): The model to train.
        train_x (list): The training data (input).
        train_y (list): The training data (output).
        epochs (int): The number of epochs to train for.
        batch_size (int): The batch size for training.
        model_path (str): The path to save the trained model.
    """
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(model_path, hist)
    print("Model is saved to", model_path)

def save_data(words, classes, words_path, classes_path):
    """
    Save words and classes to files using pickle.

    Args:
        words (list): A list of words.
        classes (list): A list of classes.
        words_path (str): The path to save the words.
        classes_path (str): The path to save the classes.
    """
    pickle.dump(words, open(words_path, 'wb'))
    pickle.dump(classes, open(classes_path, 'wb'))

def main():
    intents = load_intents('intents.json')
    words, classes, documents = preprocess_data(intents)
    training, train_x, train_y = create_training_data(words, classes, documents)
    model = build_model(len(train_x[0]), len(train_y[0]))
    train_and_save_model(model, train_x, train_y, epochs=200, batch_size=5, model_path='chatbot_model.keras')
    save_data(words, classes, 'words.pkl', 'classes.pkl')
    print("Training data created, model trained and saved, data saved.")

if __name__ == "__main__":
    main()
