import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import correlations.Utilities as utilities


def load_and_preprocess_data(dataset_path):
    # Load data
    df = pd.read_csv(dataset_path)
    df = df.dropna(axis=0)
    predictors = [2, 6, 10, 14, 18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78,82,86,90,94,98,102,106,110,114,118,122,126,130,134]
    X = df.take(predictors, axis=1)  # predictors
    X = X.iloc[1:].values

    df = utilities.unify_classes(df)
    Y = df['mood'] + "_" + df['activity'] + "_" + df['period']  # predicted_class
    Y = Y.iloc[1:]
    print("Y.head = ", Y.head())

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print("x_train.shape = ",x_train.shape)
    print("x_test.shape = ",x_test.shape)

    # Preprocess the data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # scale the input image pixels to the range [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    classes = np.unique(Y.values)
    num_classes = len(classes)
    print("We have {} unique classes: {}".format(num_classes, classes))

    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    # Transform the labels into vectors in the range [0, num_classes]
    # generate a vector for each label where the index of the label
    # is set to `1` and all other entries to `0`
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("y_train.shape = ",y_train.shape)
    print("y_test.shape = ",y_test.shape)

    return (x_train, y_train), (x_test, y_test)

#Build the Feed forward model
def build_and_compile_model(optimizer = 'adam', learn_rate=0.01, momentum=0.8):
    model = Sequential()
    model.add(Dense(484, activation='relu', input_shape=(34,))) # 34 features
    model.add(Dense(484, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(484, activation='relu'))
    model.add(Dense(27, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

# evaluate the network
def model_eval(history, epochs, img_path):
    print("Evaluating network...")

    # plot the training and test loss and accuracy
    N = np.arange(0, epochs,1)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="test_loss")
    plt.plot(N, history.history["accuracy"], label="train_acc")
    plt.plot(N, history.history["val_accuracy"], label="test_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(img_path)

def train(x_train, x_test, y_train, y_test):
    decay_rate = 0.01 / 20
    optimizer = SGD(lr=0.01, momentum=0.8, decay=decay_rate, nesterov=False)
    model=build_and_compile_model(optimizer)
    history = model.fit(x_train, y_train, epochs=20, batch_size=128,validation_data=(x_test, y_test), shuffle=True, verbose=1)
    return history

def classify(dataset_path):
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(dataset_path)
    history = train(x_train, x_test, y_train, y_test)
    model_eval(history, 20, "correlations/figures/youtube.png")