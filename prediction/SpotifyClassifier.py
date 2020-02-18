import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import prediction.Utilities as utilities

FIGURES_PATH = "prediction/figures/"

def preprocess_datasheet(df, users=[]):
    df['user_id'] = df['user_id'].astype(int).astype(str)
    if (utilities.particularUsersAskedFor(users)):
        # ask for correlations of specific user
        df = df[df['user_id'].isin(users)]
    df = df.dropna(axis=0)

    loudness = df.take([3], axis=1).values
    df['loudness'] = preprocessing.MinMaxScaler().fit_transform(loudness)
    return df

def spotify_generate_predictions(df):
    X = df.take([1,5,6,9,10,12,18,21], axis=1)  # predictors
    X = X.apply(pd.to_numeric)
    X = X.iloc[1:]

    df = utilities.unify_classes(df)
    df['Class'] = df['mood'] + "_" + df['activity'] + "_" + df['period']
    Y = df['Class']  # predicted_class
    Y = Y.iloc[1:]

    classes = np.unique(df['Class'].values)
    print("We have {} unique classes: {}".format(len(classes), classes))

    seed = 7
    num_trees = 30
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    predictions = model_selection.cross_val_predict(model, X, Y.values.ravel(), cv=kfold)

    accuracy = accuracy_score(Y.values.ravel(), predictions)
    print("accuracy = ", accuracy)

    return predictions, Y

def classify(dataset_path, users=[]):
    df = pd.read_excel(dataset_path)
    df = preprocess_datasheet(df, users)
    Y_predicted, Y_real = spotify_generate_predictions(df)
    if (utilities.allUsersAskedFor(users)):
        utilities.plot_confusion_matrix(Y_real.values.ravel(), Y_predicted, FIGURES_PATH+"spotify_heatmap_allUsers.png")
    elif (utilities.particularUsersAskedFor(users)):
        utilities.plot_confusion_matrix(Y_real.values.ravel(), Y_predicted, FIGURES_PATH+"spotify_heatmap_users"+str(users)+".png")
