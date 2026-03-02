from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from Recipe import *
import os
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import  MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import pickle
import ast
import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def buildCNNModel():
    global classifier
    if os.path.exists('model/1model.json'):
        with open('model/1model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/1model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/1history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        textarea.insert(END,"CNN training process completed with Accuracy = "+str(accuracy))
    else:
        encoding_dim = 32
        X_train = np.load('model/X.txt.npy')
        Y_train = np.load('model/Y.txt.npy')
        X = X_train.reshape(X_train.shape[0],(64 * 64 * 3))
        print(X.shape)
        input_img = keras.Input(shape=(X.shape[1],))
        encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
        decoded = layers.Dense(Y_train.shape[1], activation='softmax')(encoded)
        autoencoder = keras.Model(input_img, decoded)
        encoder = keras.Model(input_img, encoded)
        encoded_input = keras.Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        hist = autoencoder.fit(X, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        autoencoder.save_weights('model/autoencoder_model_weights.h5')
        model_json = autoencoder.to_json()
        with open("model/autoencoder_model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/autoencoder_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/autoencoder_history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
       # acc = data['accuracy']
        #accuracy = acc[9] * 100
        #print(accuracy)
       ## Compute metrics
        accuracy = accuracy_score(Y_train, Y_pred_classes)
        f1 = f1_score(Y_train, Y_pred_classes, average='weighted')
        conf_matrix = confusion_matrix(Y_train, Y_pred_classes)
    
        print(f'Classification Accuracy: {accuracy * 100:.2f}%')
        print(f'F1-score: {f1:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)
        print('Done')

def buildvgg():
    X_train = np.load('model/X.txt.npy')
    Y_train = np.load('model/Y.txt.npy')

    X = X_train.reshape(X_train.shape[0],(64 * 64 * 3))  # Reshape input data
    Y_train_classes = np.argmax(Y_train, axis=1)  # Convert one-hot labels to class indices

    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dense(units=Y_train.shape[1], activation='softmax'))  # Output layer

    print(classifier.summary())

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist1 = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)

    # Retrieve the final training accuracy
    accuracy = hist1.history['accuracy'][-1]

    # Get predictions
    Y_pred = classifier.predict(X_train)
    Y_pred_classes = np.argmax(Y_pred, axis=1)  # Convert predicted probabilities to class indices

    # Compute accuracy, F1-score, confusion matrix and classification_report
    acc_score = accuracy_score(Y_train_classes, Y_pred_classes)
    f1 = f1_score(Y_train_classes, Y_pred_classes, average='weighted')
    conf_matrix = confusion_matrix(Y_train_classes, Y_pred_classes)
    class_report = classification_report(Y_train_classes, Y_pred_classes)
    
    # Plot training loss and accuracy
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer: Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(hist1.history['loss'], label='Training Loss')
    plt.plot(hist1.history['accuracy'], label='Training Accuracy')
    plt.legend()
    plt.show()

    # Save the model
    classifier.save('model/cnn_model_weights.h5')
    model_json = classifier.to_json()
    with open("model/cnn_model.json", "w") as json_file:
        json_file.write(model_json)

    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist1.history, f)
    f.close()

    # Print metrics
    print(f'Classification Accuracy: {acc_score * 100:.2f}%')
    print(f'F1-score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)
# Load the CSV file into a DataFrame
filename = './indian dataset_123.csv'
dataset = pd.read_csv(filename,encoding='utf-8')

recipe_list = []  # Initialize a list to store recipe objects

for i, row in dataset.iterrows():
    r_name = row['TranslatedRecipeName']
    ingredients = row['Cleaned-Ingredients']
    cooking = row['TranslatedInstructions']#ast.literal_eval(row['TranslatedInstructions'])#.get('directions')
    r_name = r_name.strip().lower()
    
    obj = Recipe()

    obj.setName(r_name)
    obj.setIngredients(ingredients)
    obj.setCooking(cooking)
    
    recipe_list.append(obj)
recipe_lists = np.asarray(recipe_list)
np.save("index.txt.npy",recipe_lists)
indian = np.load('index.txt.npy', allow_pickle=True)
for i in range(len(indian)):
    recipe_list.append(indian[i])

obj = recipe_list[-1]
print(obj.getName())
encoding_dim = 32
buildvgg()
