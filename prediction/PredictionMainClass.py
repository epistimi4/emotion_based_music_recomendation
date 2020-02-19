import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import prediction.YoutubeNeuronNetwork as ffn
import prediction.SpotifyClassifier as sc

SPOTIFY_DATASET_PATH = './Business Analysis/data/spotify_metadata_playlist.xlsx'
YOUTUBE_DATASET_PATH = './Business Analysis/data/youtube_metadata_per_user.csv'

def main(platform='spotify', users=[]):
    if(platform == 'spotify'):
        sc.classify(SPOTIFY_DATASET_PATH, users)
    elif(platform == 'youtube'):
        ffn.classify(YOUTUBE_DATASET_PATH, users)
        print("Add youtube classifier")

# Predict regarding all users for spotify: PredictionMainClass.py -p spotify
# Predict regarding user 16 for spotify: PredictionMainClass.py -p spotify -u 16
# Predict regarding all users for youtube: PredictionMainClass.py -p youtube
# Predict regarding user 16 for youtube: PredictionMainClass.py -p youtube -u 16
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', '-p', help="spotify or youtube", type=str)
    parser.add_argument('--users', '-u', help="id(s) of users separated by commas", type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    platform = args.platform #spotify or youtube
    users = str(args.users).split(',') #for all users the empty parameter is translated to ['0']
    main(platform, users)
