import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import prediction.Utilities as utilities

FIGURES_PATH = "prediction/figures/"
le = LabelEncoder()

def load_and_preprocess_data(dataset_path,users=[]):
    # Load data
    df = pd.read_csv(dataset_path)
    df['user_id'] = df['user_id'].astype(int).astype(str)
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

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print("x_train.shape = ",x_train.shape)
    print("x_test.shape = ",x_test.shape)

    # Preprocess the data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # scale the input variables to the range [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    classes = np.unique(Y.values)
    num_classes = len(classes)
    print("We have {} unique classes: {}".format(num_classes, classes))

    # encode the labels, converting them from strings to integers

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
#import kfold validation by sklearn into keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
def build_classifier():
    model = Sequential()
    model.add(Dense(484, activation='relu', input_shape=(34,)))  # 34 features
    model.add(Dense(484, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(484, activation='relu'))
    model.add(Dense(6, activation='softmax'))  # when run for user 1, should have output=9
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(x_train, x_test, y_train, y_test):
    classifier = KerasClassifier(build_fn=build_classifier, epochs=20, batch_size=128)
    # predictions = cross_val_predict(estimator=classifier, X=x_train, y=y_train, cv=10)

    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=3)
    mean = accuracies.mean() #we want one output accuracy, so we calculate the mean
    variance = accuracies.std()
    print("The mean accuracy is {} and the variance is {}".format(mean, variance))



def classify(dataset_path, users=[]):
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(dataset_path, users)
    train(x_train, x_test, y_train, y_test)
    # history = train(x_train, x_test, y_train, y_test)
    # if (utilities.allUsersAskedFor(users)):
    #     model_eval(history, 20, "youtube_accuracy_allUsers")
    # elif (utilities.particularUsersAskedFor(users)):
    #     model_eval(history, 20, "youtube_accuracy_users" + str(users))
