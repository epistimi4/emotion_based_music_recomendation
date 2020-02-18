import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix


def unify_classes(df):
    # The parent mood unhappy contains the moods angry,sad and nervous
    df['mood'] = df['mood'].replace({'angry': 'unhappy', 'sad': 'unhappy', 'nervous': 'unhappy'})
    # The parent mood annoyed contains the moods bored and sleepy
    df['mood'] = df['mood'].replace({'bored': 'annoyed', 'sleepy': 'annoyed'})
    # The parent mood relaxed contains the moods calm,relaxed and peaceful
    df['mood'] = df['mood'].replace({'calm': 'relaxed', 'relaxed': 'relaxed', 'peaceful': 'relaxed'})
    # The parent mood happy contains the moods excited,happy and pleased
    df['mood'] = df['mood'].replace({'excited': 'happy', 'happy': 'happy', 'pleased': 'happy'})

    # The parent period midday contains the periods evening and afternoon
    df['period'] = df['period'].replace({'evening': 'midday', 'afternoon': 'midday'})

    df['activity'] = df['activity'].replace({'commuting': 'other'})
    return df

def plot_confusion_matrix(Y, predictions,path):
    cf_matrix = confusion_matrix(predictions, Y.values.ravel())
    classes = np.unique(Y.values)
    fig = plt.figure(figsize=(50,50))
    sn.set(font_scale=1.5)
    mood_matx = sn.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, xticklabels=classes, yticklabels=classes,
                           fmt='.2%', cmap='Reds')
    mood_matx.set_xticklabels(mood_matx.get_xticklabels(), rotation=90)
    plt.xlabel('Predicted Class')
    plt.ylabel('Real Class')
    fig.savefig(path)

def allUsersAskedFor(users=['0']):
    return len(users) > 0 and users[0] == '0'

def particularUsersAskedFor(users=['0']):
    return len(users) > 0 and users[0] != '0'