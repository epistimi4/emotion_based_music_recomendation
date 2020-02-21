import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder

import prediction.Utilities as utilities

FIGURES_PATH = "prediction/figures/"
le = LabelEncoder()

def load_and_preprocess_data(dataset_path,users=[]):
    # Load data
    df = pd.read_csv(dataset_path)
    if (utilities.particularUsersAskedFor(users)):
        # ask for songs of specific user
        df = df[df['user_id'].isin(users)]

    df = df.dropna(axis=0)

    num_users = len(np.unique(df['user_id'].values))
    print("{} songs listened by {} users".format(df.shape[0]-1, num_users))

    predictors = [2, 6, 10, 14, 18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78,82,86,90,94,98,102,106,110,114,118,122,126,130,134]
    X = df.take(predictors, axis=1)  # predictors
    X = X.iloc[1:].values

    df = utilities.unify_classes(df)
    Y = df['mood'] + "_" + df['activity'] + "_" + df['period']  # predicted_class
    Y = Y.iloc[1:]
    print("Y.head = ", Y.head())

    # Preprocess the data
    X = X.astype('float32')

    # scale the input variables to the range [0, 1]
    X = X / 255.0

    classes = np.unique(Y.values)
    num_classes = len(classes)
    print("We have {} unique classes: {}".format(num_classes, classes))

    # encode the labels, converting them from strings to integers
    Y = le.fit_transform(Y)

    # Transform the labels into vectors in the range [0, num_classes]
    # generate a vector for each label where the index of the label
    # is set to `1` and all other entries to `0`
    Y = keras.utils.to_categorical(Y, num_classes)

    return X,Y,classes

def build_classifier(output_dim = 34):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(34,)))  # 34 features
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))  # when run for user 1, should have output=9
    optimizer = optimizers.SGD(lr=0.01, momentum=0.8, nesterov=False)
    model.compile(loss=[focal_loss], metrics=["accuracy"], optimizer=optimizer)
    return model

 # Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def train_evaluate(X,Y, users, classes):
    epochs = 50
    classifier = KerasClassifier(build_fn=build_classifier, output_dim = Y.shape[1], epochs=epochs, batch_size=128)
    k = 9
    y_pred = cross_val_predict(classifier, X, Y, cv=k)
    rounded_labels = np.argmax(Y, axis=1)

    classification_report = metrics.classification_report(rounded_labels, y_pred, target_names=classes)
    print("classification_report ", classification_report)
    balanced_accuracy = metrics.balanced_accuracy_score(rounded_labels, y_pred)
    print("The balanced accuracy is ", balanced_accuracy)

    heatmap_save_path = FIGURES_PATH + "youtube_heatmap_allUsers.png"
    if (utilities.particularUsersAskedFor(users)):
        heatmap_save_path = FIGURES_PATH + "youtube_heatmap_user"+str(users)+".png"
    utilities.plot_confusion_matrix(rounded_labels, y_pred, classes, heatmap_save_path, "Greys", (50,50))


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

def exploratory_nalysis(df, users):
    figure_name = FIGURES_PATH+"youtube_class_support_allUsers.png"
    if (utilities.particularUsersAskedFor(users)):
        figure_name = FIGURES_PATH + "youtube_class_support_user"+str(users)+".png"
    fig = plt.figure(figsize=(14,18))
    df['Class'].value_counts().nlargest(40).plot(kind='bar', color='grey', edgecolor='black', linewidth=1)
    plt.title("Support for all classes")
    plt.ylabel("Number of instances")
    plt.xlabel("Class")
    fig.savefig(figure_name)
    print(df.groupby('Class').size())

def classify(dataset_path, users=[]):
    x, y, classes = load_and_preprocess_data(dataset_path, users)
    train_evaluate(x,y, users, classes)

