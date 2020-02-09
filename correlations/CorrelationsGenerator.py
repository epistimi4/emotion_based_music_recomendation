import pandas as pd
import numpy as np
import sys
import argparse
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

DATASET_PATH = '/home/maria/projects/emotion_based_music_recomendation/correlations/dataset.csv'

def main(platform='spotify', users=[]):
    df = pd.read_csv(DATASET_PATH)
    df = preprocess_datasheet(df, users)
    if(platform == 'spotify'):
        spotify_correlations(df)
    elif(platform == 'youtube'):
        print("Add youtube classifier")

def spotify_correlations(df):
    X = df.take([1, 2, 3, 4, 5, 6, 7], axis=1)  # predictors
    X = X.apply(pd.to_numeric)
    Y = df.take([9], axis=1)  # predicted_class

    seed = 7
    num_trees = 30
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    predictions = model_selection.cross_val_predict(model, X, Y.values.ravel(), cv=kfold)
    accuracy = accuracy_score(Y.values.ravel(), predictions)
    print("accuracy = ", accuracy)


def preprocess_datasheet(df, users=[]):
    df['User'] = df['User'].astype(int).astype(str)
    if(len(users)>0 and users[0] != '0'):
        # ask for correlations of specific user
        df = df[df['User'].isin(users)]
    df = df.dropna(axis=0)

    loudness = df.take([3], axis=1).values
    df['spot_loudness'] = preprocessing.MinMaxScaler().fit_transform(loudness)

    classes = np.unique(df['Class'].values)
    print("We have {} unique classes: {}".format(len(classes), classes))

    df = df.iloc[1:]
    return df

# Predict considering all users: CorrelationsGenerator.py -p spotify
# Predict considering user 1: CorrelationsGenerator.py -p spotify -u 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', '-p', help="spotify or youtube", type=str)
    parser.add_argument('--users', '-u', help="id(s) of users", type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    platform = args.platform #spotify or youtube
    users = str(args.users).split(',') #for all users the users parameter is ['0']
    main(platform, users)
