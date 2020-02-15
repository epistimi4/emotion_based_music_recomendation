from youtube_api import YouTubeDataAPI
from pytube import YouTube
import pandas as pd
import os

song_counter=0
api_key = 'AIzaSyAwSt2zLQbVIdy9Hf8adhRzuw1fks3cHr0'
yt = YouTubeDataAPI(api_key)

def find_all_files():
    path = "C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data"
    text_files = [f for f in os.listdir(path) if f.startswith('user')]
    return text_files


def retrieve_song_titles_template1(filename):
    dfs = pd.read_excel(filename, sheet_name='Sheet1')
    return dfs['song-artist']

def retrieve_song_titles_template2(filename):
    return

def get_video_metadata(song):
    try:
        searches = yt.search(q=song,
                     max_results=1)
        if len(searches) != 0:
            video_id = searches[0].get('video_id', 0)
            #yt.get_video_metadata_gen(searches[0]['video_id'])
            #print(yt.get_video_metadata(searches[0]['video_id']))
            return searches[0]['video_id']
        print("lossing song... ", song)
    except:
        print("lossing song... ", song)
        return 0

def get_video_id(song):
    try:
        searches = yt.search(q=song,
                     max_results=1)
        return searches[0]['video_id']
    except:
        print("Missing song ", song)
        return 0

def download_song(video_id, song):
    global song_counter
    song_counter = song_counter +1
    url = "https://www.youtube.com/watch?v="+video_id
    try:
        yt = YouTube(url=url)
        t = yt.streams.filter(only_audio=True, file_extension='mp4').all()
        t[0].download("E:\\songs", song )
    except:
        print(str(song_counter)+" "+url)

def songDoesNotExist(song):
    path = "E:\songs"
    text_files = [f for f in os.listdir(path) if f.startswith(song)]
    return len(text_files) == 0

text_files = find_all_files()
for file in text_files:
    songs = retrieve_song_titles_template1("C:\\Users\\epistimi\\Documents\\DataSience\\Multimodal\\emotion_based_music_recomendation\\Business Analysis\\data\\"+file)
    for song in songs:
        if songDoesNotExist(song):
            video_id = get_video_id(song)
            if video_id != 0:
                download_song(video_id, song)
#get_video_metadata('My Angel-Stive Morgan')