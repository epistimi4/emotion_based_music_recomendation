import numpy as np
import pandas
#import plotly
#import plotly.graph_objs as go
import os
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioBasicIO


n_harmonic_feats = 0
n_mfcc_feats = 13
n_chroma_feats = 13
new_feature_names = ["song-artist"]
feature_names = ["zcr", "energy", "energy_entropy"]
feature_names += ["spectral_centroid", "spectral_spread"]
feature_names.append("spectral_entropy")
feature_names.append("spectral_flux")
feature_names.append("spectral_rolloff")
feature_names += ["mfcc_{0:d}".format(mfcc_i)
                  for mfcc_i in range(1, n_mfcc_feats + 1)]
feature_names += ["chroma_{0:d}".format(chroma_i)
                  for chroma_i in range(1, n_chroma_feats)]
feature_names.append("chroma_std")
for feature_name in feature_names:
    new_feature_names.append(feature_name+"_mean")
    new_feature_names.append(feature_name + "_std")
    new_feature_names.append(feature_name + "_min")
    new_feature_names.append(feature_name + "_max")


def find_all_files():
    path = "E:\\song_conv2"
    music_files = [f for f in os.listdir(path) if f.endswith('wav')]
    return music_files


def extract_features(song_title, row_count):
    global df
    # read machine sound
    fs, s = audioBasicIO.read_audio_file(song_title)
    duration = len(s) / float(fs)
    s = audioBasicIO.stereo_to_mono(s)
    # extract short term features
    df['song-artist'][row_count] = song_title.split('.')[0]
    [f, f_names] = ShortTermFeatures.feature_extraction(s, fs, 2*fs, fs)#0.050 * fs, 0.025 * fs)
    counter = 0
    #for every vector find mean, std, min, max
    for feature in f:
        #print(np.mean(feature))
        #df[row_count][f_names[counter] + "_mean"].append(1)
        df[f_names[counter]+"_mean"][row_count] = np.mean(feature)
        df[f_names[counter] +"_std"][row_count] = np.std(feature)
        df[f_names[counter] +"_min"][row_count] = min(feature)
        df[f_names[counter] +"_max"][row_count] = max(feature)
        counter = counter +1
    return 0

files = find_all_files()
index = range(len(files))
df = pandas.DataFrame(index = index, columns = new_feature_names)
#df = df.fillna(0)
counter = 0
for file in files:
    extract_features("E:\\song_conv2\\" + file, counter)
    counter = counter + 1
df.to_csv("C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data\\you_tube_metadata.csv")


