import json
import string
import random

import joblib
import nltk
import numpy as np

from keras.layers import Dropout
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, MaxPool2D, Conv2D, Flatten
import matplotlib.pyplot as plt

#nltk.download("punkt")
#nltk.download("wordnet")

data = json.loads(open('intents.json').read())

# Creating data_X and data_y
words = []  # For FIFAT model/ vocabulary for patterns
classes = []  # For FIFAT model/ vocabulary for tags
data_X = []  # For storing each pattern
data_y = []  # for storing tag corresponding to each pattern in data_X

# Iterating over all the intents
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)  # tokenize each pattern
        words.extend(tokens)  # and append tokens to words
        data_X.append(pattern)  # appending pattern to data X
        data_y.append(intent["tag"]),  # appending the associated tag to each pattern

    # adding the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()
# lemmatize all the words in the vocabulary and convert them to lowercase
# if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and takin the # set to ensure no duplicates occur

words = sorted(set(words))
classes = sorted(set(classes))

# Words to Numbers
training_data_set = []
out_empty = [0] * len(classes)
# developinng the bag of words for model
for idx, doc in enumerate(data_X):
    bag_of_word = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bag_of_word.append(1) if word in text else bag_of_word.append(0)
    # mark the index of class that the current pattern is associated to
    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1
    # add the one hot encoded BoW and associated classes to training
    training_data_set.append([bag_of_word, output_row])
# shuffle the  data and convert it to an array
random.shuffle(training_data_set)

#training_data, testing_data = train_test_split(training, test_size=0.2, random_state=25)

#print(f"No. of training examples: {training}")
#print(f"No. of testing examples: {training}")

arr = np.random.rand(100, 3)
training_data_set = np.array(training_data_set, dtype=object)
spl = 0.7
N = len(arr)
sample = int(spl*N)

train_idx, test_idx = training_data_set[:sample], training_data_set[sample:]

print(f"training data : {train_idx}")
print(f"test data : {test_idx}")

# split the features and target labels
train_X = np.array(list(training_data_set[:, 0]))
train_Y = np.array(list(training_data_set[:, 1]))

# The Neural Network Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu")) #relu optionn is followed and pass previous layer data to other layers
model.add(Dropout(0.5)) #boundary
model.add(Dense(64, activation="relu")) #number of neurons
model.add(Dropout(0.5)) #boundary
model.add(Dense(len(train_Y[0]), activation="softmax"))
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6) #ML algorithm
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
print(model.summary())
history = model.fit(x=train_X,y=train_Y, epochs=150, verbose=1)

print(history.history['loss'])
print(history.history['accuracy'])
# plt.plot(history.history['loss'])
# plt.title("Model  loss")
# plt. ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(['train'],loc="upper left")
# plt.show()

# plt.plot(history.history['accuracy'])
# plt.title("Model accuracy ")
# plt. ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(['train'],loc="upper left")
# plt.show()

#model.save('fifatmodel.model')
joblib.dump(model, 'fifatmodel.model')
joblib.dump(words,'words.joblib')
joblib.dump(classes,'classes.joblib')
# Preprocessing the Input
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bag_of_word = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bag_of_word[idx] = 1
    return np.array(bag_of_word)



