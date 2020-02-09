import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials #to access authorised Spotify data
import json
import csv
import glob

MY_CLIENT_ID='5076fb68a207470981187d66a6e534a8'
MY_CLIENT_SECRET='3a59a64b8d0d470cb594fb2d5ac1d600'
F_PATTERN = '/home/maria/projects/emotion_based_music_recomendation/Business Analysis/data/Template*.csv'
DATASET_PATH = '/home/maria/projects/emotion_based_music_recomendation/correlations/dataset.csv'

client_credentials_manager = SpotifyClientCredentials(client_id=MY_CLIENT_ID, client_secret=MY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

class Row(object):
    def __init__(self, song,artist,mood,activity,location):
        self.song=song
        self.artist = artist
        self.mood = mood
        self.activity = activity
        self.location = location
        self.label = mood.lower()+"_at_"+location.lower()+"_while_"+activity.lower()

    def __str__(self):
        return f'{self.song}, {self.artist}, {self.mood}, {self.activity}, {self.location}'

def process_file(filename):
    rows=[]
    with open(filename) as fp:
        next(fp)
        line = fp.readline()
        while line:
            fields=line.strip().split(',')
            song_artist=fields[2].split('-')
            mood = fields[3]
            location = fields[5]
            activity = fields[6]
            row=Row(song_artist[1],song_artist[0],mood,activity,location)
            rows.append(row)
            line = fp.readline()
    return rows

def parse_input_files():
    filenames = [filename for filename in sorted(glob.glob(F_PATTERN))]
    print(filenames)
    dict = {}
    user_id = 1
    for filename in filenames:
        processed_rows = process_file(filename)
        dict.update({user_id: processed_rows})
        user_id = user_id + 1
    return dict

def retrieve_song_features(user_history):
    tracks=[]
    valid_personal_metadata=[]
    for row in user_history:
        result = queryTrack(row.artist, row.song)
        if len(result['tracks']['items']) > 0:
            if(result['tracks']['items'][0]['id']!=None):
                track_id = result['tracks']['items'][0]['id']
                print(track_id)
                tracks.append(track_id)
                valid_personal_metadata.append(row)
        else:
            print('song {0} not found'.format(row.song))
    tracks_metadata=queryTrackMetadata(tracks)
    audio_features = map_to_audio_features(tracks_metadata)
    return audio_features,valid_personal_metadata

def queryTrack(artist,track):
    q = "artist:{} track:{}".format(artist, track)
    track=sp.search(q=q, type="track", limit=10)
    return track

def queryTrackMetadata(ids):
    json_metadata = sp.audio_features(ids)
    metadata_list = json.loads(str(json_metadata).replace("'","\""))
    return metadata_list

def map_to_audio_features(metadata_list=[]):
    audio_features = {}
    danceability_values = []
    energy_values = []
    loudness_values = []
    speechiness_values = []
    acousticness_values = []
    instrumentalness_values = []
    liveness_values = []
    valence_values = []
    for i in range(0, len(metadata_list) - 1):
        song_metadata = metadata_list[i]
        danceability_values.append(song_metadata.get("danceability"))
        energy_values.append(song_metadata.get("energy"))
        loudness_values.append(song_metadata.get("loudness"))
        speechiness_values.append(song_metadata.get("speechiness"))
        acousticness_values.append(song_metadata.get("acousticness"))
        instrumentalness_values.append(song_metadata.get("instrumentalness"))
        liveness_values.append(song_metadata.get("liveness"))
        valence_values.append(song_metadata.get("valence"))
    audio_features.update({'danceability': danceability_values})
    audio_features.update({"energy": energy_values})
    audio_features.update({"loudness": loudness_values})
    audio_features.update({"speechiness": speechiness_values})
    audio_features.update({"acousticness": acousticness_values})
    audio_features.update({"instrumentalness": instrumentalness_values})
    audio_features.update({"liveness": liveness_values})
    audio_features.update({"valence": valence_values})
    return audio_features

def create_dataset(personal_metadata, spotify_metadata):
    with open(DATASET_PATH, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["User", "spot_danceability", "spot_energy", "spot_loudness", "spot_speechiness", "spot_acousticness", "spot_instrumentalness",
                 "spot_liveness", "spot_valence", "Class"])

        for user_id in personal_metadata:
            spotify_metadata_tracks = spotify_metadata.get(user_id)
            danceability_values = spotify_metadata_tracks.get('danceability')
            energy_values = spotify_metadata_tracks.get("energy")
            loudness_values = spotify_metadata_tracks.get("loudness")
            speechiness_values = spotify_metadata_tracks.get("speechiness")
            acousticness_values = spotify_metadata_tracks.get("acousticness")
            instrumentalness_values = spotify_metadata_tracks.get("instrumentalness")
            liveness_values = spotify_metadata_tracks.get("liveness")
            valence_values = spotify_metadata_tracks.get("valence")
            personal_metadata_rows = personal_metadata.get(user_id)
            for i in range(0, len(personal_metadata_rows)-1):
                personal_metadata_row = personal_metadata_rows[i]
                filewriter.writerow([user_id, danceability_values[i], energy_values[i], loudness_values[i], speechiness_values[i], acousticness_values[i], instrumentalness_values[i],liveness_values[i], valence_values[i], personal_metadata_row.label])

if __name__== "__main__":
    personal_metadata=parse_input_files() # get the songs for all users in a dictionary {user_id:Row_list}
    spotify_metadata = {}
    for user_id in personal_metadata:
        spotify_metadata_user, personal_metadata_user = retrieve_song_features(personal_metadata.get(user_id))  # get audio features for the songs of each user in a dictionary
        personal_metadata.update({user_id : personal_metadata_user})
        print(spotify_metadata_user)
        spotify_metadata.update({user_id : spotify_metadata_user})
    create_dataset(personal_metadata,spotify_metadata)






