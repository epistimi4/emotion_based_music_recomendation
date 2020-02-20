import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn import model_selection, preprocessing
from sklearn.ensemble import AdaBoostClassifier

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

    df = utilities.unify_classes(df)
    df['Class'] = df['mood'] + "_" + df['activity'] + "_" + df['period']
    return df

def balancedClassifier(df):
    # Create an object of the classifier.
    seed = 7
    num_trees = 30
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    base_estimator = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    ee_classifier = EasyEnsembleClassifier(n_estimators=10,
                                 base_estimator=base_estimator)

    X = df.take([1, 5, 6, 9, 10, 12, 18, 21], axis=1)  # predictors
    X = X.apply(pd.to_numeric)
    X = X.iloc[1:]

    Y = df['Class']  # predicted_class
    Y = Y.iloc[1:]

    classes = np.unique(df['Class'].values)
    print("We have {} unique classes: {}".format(len(classes), classes))

    # Train the classifier.
    ee_classifier.fit(X, Y)
    predictions = model_selection.cross_val_predict(ee_classifier, X, Y.values.ravel(), cv=kfold)
    classification_report = metrics.classification_report(Y.values.ravel(), predictions, target_names=classes)
    print("classification_report ",classification_report)
    balanced_accuracy = metrics.balanced_accuracy_score(Y.values.ravel(), predictions)
    print(" Balanced accuracy = ", balanced_accuracy)
    return predictions, Y

def exploratory_nalysis(df):
    print("Summarise distribution of instances across classes", df.groupby('Class').size())

def classify(dataset_path, users=[]):
    df = pd.read_excel(dataset_path)
    df = preprocess_datasheet(df, users)
    exploratory_nalysis(df)
    # The dataset is unbalanced. We cannot use SMOTE(there are classes with one instance)
    Y_predicted, Y_real = balancedClassifier(df)
    if (utilities.allUsersAskedFor(users)):
        utilities.plot_confusion_matrix(Y_real.values.ravel(), Y_predicted, FIGURES_PATH+"spotify_heatmap_allUsers.png")
    elif (utilities.particularUsersAskedFor(users)):
        utilities.plot_confusion_matrix(Y_real.values.ravel(), Y_predicted, FIGURES_PATH+"spotify_heatmap_users"+str(users)+".png")
