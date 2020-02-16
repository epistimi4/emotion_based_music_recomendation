import os
import pandas as pd



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
feature_names.append("gender")
feature_names.append("age")
feature_names.append("location")
feature_names.append("mood")
feature_names.append("activity")
feature_names.append("period")
feature_names.append("user_id")

def find_all_user_files():
    path = "C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data"
    text_files = [f for f in os.listdir(path) if f.startswith('user')]
    return text_files

def find_all_raw_data_files():
    path = "C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data"
    text_files = [f for f in os.listdir(path) if f.startswith('you_tube_metadata')]
    return text_files


def retrieve_song_titles_template1(filename):
    dfs = pd.read_excel(filename, sheet_name='Sheet1')
    return dfs['song-artist']

df_final = pd.DataFrame(index = range(1000), columns = new_feature_names)
row_counter = 0
df = pd.DataFrame()
text_files = find_all_raw_data_files()
for file in text_files:
    if df.empty:
        df = pd.read_csv("C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data\\"+file)
    else:
        df.append(pd.read_csv(
            "C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data\\" + file))
user_files = find_all_user_files()
for file in user_files:
    user_file_row = 0
    data_available = False
    df_user = pd.read_excel("C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data\\"+file)
    for value in df_user['song-artist']:
        result = df.loc[df['song-artist'] == value]
        #result.drop('Unnamed: 0')
        for column in result.columns:
            if column in new_feature_names and len(result[column].values) > 0:
                df_final.at[row_counter, column] = result[column].values[0]
                data_available = True
        if data_available:
            df_final.at[row_counter, 'gender'] = df_user.at[user_file_row, 'gender']
            df_final.at[row_counter, 'age'] = df_user.at[user_file_row, 'age']
            df_final.at[row_counter, 'location'] = df_user.at[user_file_row, 'location']
            df_final.at[row_counter, 'mood'] = df_user.at[user_file_row, 'mood']
            df_final.at[row_counter, 'activity'] = df_user.at[user_file_row, 'activity']
            df_final.at[row_counter, 'period'] = df_user.at[user_file_row, 'period']
            df_final.at[row_counter, 'user_id'] = df_user.at[user_file_row, 'Id']
            row_counter = row_counter + 1
        user_file_row = user_file_row + 1
df_final.to_csv("C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data\\youtube_metadata_per_user.csv")

